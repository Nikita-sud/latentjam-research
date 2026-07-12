"""Export the history-aware predictor to ONNX (state encoder + scorer).

Two artifacts ship to the Android runtime:

* ``predictor_state.onnx`` — wraps ``HistoryAwareWithScorer.encode_state``.
  Inputs are the five feature tensors the recommender builds at request
  time; output is a 512-d L2-normalized state vector. Dynamic batch.

* ``predictor_scorer_n<N>.onnx`` — wraps ``HistoryAwareWithScorer.score``
  with the candidate count pinned at N (default 100). Mobile reranks the
  top-N from the cosine retrieval pass; libraries with fewer than N
  embedded tracks zero-pad and mask padded scores to ``-inf`` on-device.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
import torch
from torch import nn

from predictor.history_predictor import (
    HistoryAwarePredictor,
    HistoryAwareWithScorer,
    HistoryPredictorConfig,
    SCORING_PREDICTOR_VERSION,
    ScoringHead,
    ScoringHeadConfig,
)


def _load_predictor(checkpoint: Path) -> tuple[HistoryAwareWithScorer, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_kwargs = {
        k: v for k, v in state["state_config"].items() if k != "predictor_version"
    }
    # Older checkpoints predate `use_played_pct`; default to False to avoid an
    # unwanted `token_in_proj` whose weights aren't in the state dict.
    state_kwargs.setdefault("use_played_pct", False)
    state_cfg = HistoryPredictorConfig(**state_kwargs)
    score_cfg = ScoringHeadConfig(
        **{k: v for k, v in state["scoring_config"].items() if k != "predictor_version"}
    )
    model = HistoryAwareWithScorer(state_cfg, score_cfg)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    meta = {
        "predictor_version": state.get("predictor_version", SCORING_PREDICTOR_VERSION),
        "state_config": asdict(state_cfg),
        "scoring_config": asdict(score_cfg),
    }
    return model, meta


class _StateEncoderWrapper(nn.Module):
    def __init__(self, predictor: HistoryAwarePredictor):
        super().__init__()
        self.predictor = predictor

    def forward(
        self,
        history_small: torch.Tensor,
        history_medium: torch.Tensor,
        history_large: torch.Tensor,
        time_features: torch.Tensor,
        session_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictor(
            history_small, history_medium, history_large, time_features, session_features
        )


class _ScorerWrapper(nn.Module):
    """Pin candidate count and skip the dim-juggling branches in `ScoringHead`.

    Takes **pre-broadcasted** state of shape (B, N, D) instead of (B, D). The
    caller is responsible for broadcasting; this is trivial in Kotlin/Python
    (it's a tensor allocation + memcpy) and lets the exported ONNX graph stay
    free of `Expand` ops, which the legacy `onnx2ncnn` and the modern PNNX
    converter both reject with "expand tensor along batch index 0 is not
    supported yet". With this change the graph is reduced to elementwise
    ops + MatMul, which every mobile inference runtime handles cleanly.
    """

    def __init__(self, head: ScoringHead, n_candidates: int):
        super().__init__()
        self.head = head
        self.n_candidates = n_candidates

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # state: (B, D) — broadcast over the N candidate slots inside the graph.
        # The app (ONNX Runtime) passes an un-broadcast (1, D) state named "state";
        # ORT handles the resulting Expand op cleanly (unlike onnx2ncnn/PNNX), so we
        # keep the mobile scorer interface matching PredictorRuntime.score().
        state_broadcast = state.unsqueeze(1).expand(-1, self.n_candidates, -1)
        # state_broadcast: (B, N, D), candidates: (B, N, D)
        prod = state_broadcast * candidates
        diff = (state_broadcast - candidates).abs()
        feats = [state_broadcast, candidates, prod, diff]
        if self.head.cfg.use_state_anchor:
            cos = (state_broadcast * candidates).sum(dim=-1, keepdim=True)
            feats.append(cos)
        x = torch.cat(feats, dim=-1)
        return self.head.mlp(x).squeeze(-1)


def export_state(model: HistoryAwareWithScorer, out_path: Path, *, opset: int = 17) -> Path:
    cfg = model.state_encoder.cfg
    d = cfg.embedding_dim
    k = cfg.context_k
    history_token_dim = d + (1 if cfg.use_played_pct else 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _StateEncoderWrapper(model.state_encoder).eval()
    dummies = (
        torch.zeros(1, k, history_token_dim),
        torch.zeros(1, d),
        torch.zeros(1, d),
        torch.zeros(1, cfg.time_feat_dim),
        torch.zeros(1, cfg.session_feat_dim),
    )
    dynamic_axes = {
        "history_small": {0: "batch"},
        "history_medium": {0: "batch"},
        "history_large": {0: "batch"},
        "time_features": {0: "batch"},
        "session_features": {0: "batch"},
        "state": {0: "batch"},
    }
    torch.onnx.export(
        wrapper,
        dummies,
        out_path.as_posix(),
        input_names=[
            "history_small",
            "history_medium",
            "history_large",
            "time_features",
            "session_features",
        ],
        output_names=["state"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        export_params=True,
        dynamo=False,
    )
    return out_path


def export_scorer(
    model: HistoryAwareWithScorer,
    out_path: Path,
    *,
    n_candidates: int,
    opset: int = 17,
) -> Path:
    score_cfg = model.scoring_head.cfg
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _ScorerWrapper(model.scoring_head, n_candidates).eval()
    dummies = (
        torch.zeros(1, score_cfg.state_dim),
        torch.zeros(1, n_candidates, score_cfg.candidate_dim),
    )
    dynamic_axes = {
        "state": {0: "batch"},
        "candidates": {0: "batch"},
        "scores": {0: "batch"},
    }
    torch.onnx.export(
        wrapper,
        dummies,
        out_path.as_posix(),
        input_names=["state", "candidates"],
        output_names=["scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        export_params=True,
        dynamo=False,
    )
    return out_path


@click.command()
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("models/predictor/scoring_v1.pt"),
    show_default=True,
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models/onnx"),
    show_default=True,
)
@click.option("--candidates", type=int, default=100, show_default=True)
@click.option("--opset", type=int, default=17, show_default=True)
def main(checkpoint_path: Path, out_dir: Path, candidates: int, opset: int) -> None:
    model, meta = _load_predictor(checkpoint_path)
    state_out = export_state(model, out_dir / "predictor_state.onnx", opset=opset)
    scorer_out = export_scorer(
        model,
        out_dir / f"predictor_scorer_n{candidates}.onnx",
        n_candidates=candidates,
        opset=opset,
    )
    cfg = model.state_encoder.cfg
    sc = model.scoring_head.cfg
    history_token_dim = cfg.embedding_dim + (1 if cfg.use_played_pct else 0)
    meta_text = "\n".join(
        [
            f"predictor_version={meta['predictor_version']}",
            f"checkpoint={checkpoint_path}",
            f"opset={opset}",
            f"context_k={cfg.context_k}",
            f"embedding_dim={cfg.embedding_dim}",
            f"history_small_shape=(B, {cfg.context_k}, {history_token_dim})",
            f"history_medium_shape=(B, {cfg.embedding_dim})",
            f"history_large_shape=(B, {cfg.embedding_dim})",
            f"time_features_shape=(B, {cfg.time_feat_dim})",
            f"session_features_shape=(B, {cfg.session_feat_dim})",
            f"state_shape=(B, {cfg.embedding_dim})",
            f"scorer_candidates={candidates}",
            f"scorer_state_shape=(B, {sc.state_dim})",
            f"scorer_candidates_shape=(B, {candidates}, {sc.candidate_dim})",
            f"scorer_output_shape=(B, {candidates})",
        ]
    )
    (state_out.with_suffix(".meta.txt")).write_text(meta_text + "\n")
    (scorer_out.with_suffix(".meta.txt")).write_text(meta_text + "\n")
    click.echo(f"Wrote {state_out}")
    click.echo(f"Wrote {scorer_out}")


if __name__ == "__main__":
    main()
