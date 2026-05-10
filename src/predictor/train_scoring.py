"""Train HistoryAwareWithScorer (state encoder + cross-encoder scoring head).

Two heads share one state encoder:
- two-tower head (state vs candidate cosine) -- fast retrieval, supervised
  by InfoNCE on cosine similarities (same as ``train_history.py``).
- scoring head -- listwise softmax CE over (1 positive + N negatives) of
  the cross-encoder ``score(state, candidate)``. Directly optimizes top-1
  ranking on a fixed candidate set.

At inference the app can use the scoring head for re-ranking the
two-tower top-K shortlist, or score the full library if it's small.

Eval reports both retrieval modes side-by-side:

- val_neg100[two-tower]:  cosine(state, candidate) over 1+100 candidates
- val_neg100[scoring]:    scoring_head(state, candidate) over 1+100 candidates

The scoring head should match or beat the two-tower head on recall@1
because it can learn non-linear interactions between state and candidate
that pure cosine cannot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from predictor.history_predictor import (
    SCORING_PREDICTOR_VERSION,
    HistoryAwareWithScorer,
    HistoryPredictorConfig,
    ScoringHeadConfig,
)
from predictor.store import EmbeddingStore
from predictor.train_history import (
    CONTEXT_K,
    HistoryEventDataset,
    TrainPair,
    _build_pairs_for_user,
    _baseline_query,
    collate,
    info_nce,
)


# ---------------------------------------------------------------------------
# Eval (both heads side-by-side)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate_sampled_both(
    model: HistoryAwareWithScorer,
    val_pairs: list[TrainPair],
    matrix: np.ndarray,
    *,
    device: torch.device,
    n_samples: int = 2000,
    n_negatives: int = 100,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> tuple[dict[str, float], dict[str, float]]:
    """Sampled (1 positive + n_negatives random) eval for both heads."""
    rng = np.random.default_rng(rng_seed)
    n_pool = matrix.shape[0]
    if len(val_pairs) > n_samples:
        idxs = rng.choice(len(val_pairs), size=n_samples, replace=False)
    else:
        idxs = np.arange(len(val_pairs))

    pool_t = torch.from_numpy(matrix).to(device)
    model.eval()

    ranks_two_tower: list[int] = []
    ranks_scoring: list[int] = []

    for i in idxs:
        p = val_pairs[i]
        target_row = int(np.argmax(matrix @ p.target))
        neg_idx = rng.choice(n_pool, size=n_negatives + 5, replace=False)
        neg_idx = neg_idx[neg_idx != target_row][:n_negatives]
        candidate_rows = np.concatenate([[target_row], neg_idx])
        cand_emb = pool_t[candidate_rows]                            # (N+1, D)

        h_small = torch.from_numpy(p.history_small).to(device).unsqueeze(0)
        h_med = torch.from_numpy(p.history_medium).to(device).unsqueeze(0)
        h_lg = torch.from_numpy(p.history_large).to(device).unsqueeze(0)
        tf = torch.from_numpy(p.time_feat).to(device).unsqueeze(0)
        sf = torch.from_numpy(p.session_feat).to(device).unsqueeze(0)

        state = model.encode_state(h_small, h_med, h_lg, tf, sf).squeeze(0)  # (D,)

        # Two-tower: cosine.
        scores_tt = cand_emb @ state                                  # (N+1,)
        order = torch.argsort(scores_tt, descending=True)
        rank_tt = int((order == 0).nonzero(as_tuple=False).item()) + 1
        ranks_two_tower.append(rank_tt)

        # Scoring head: cross-encoder on the same candidate set.
        scores_sc = model.score(state.unsqueeze(0), cand_emb.unsqueeze(0)).squeeze(0)  # (N+1,)
        order = torch.argsort(scores_sc, descending=True)
        rank_sc = int((order == 0).nonzero(as_tuple=False).item()) + 1
        ranks_scoring.append(rank_sc)

    def _summarize(ranks: list[int], mode: str) -> dict[str, Any]:
        arr = np.asarray(ranks, dtype=np.int64)
        out = {
            "n_samples": int(len(arr)),
            "mrr": float(np.mean(1.0 / np.maximum(arr, 1))),
            "median_rank": float(np.median(arr)),
            "mode": mode,
        }
        for k in ks:
            out[f"recall@{k}"] = float(np.mean(arr <= k))
        return out

    return _summarize(ranks_two_tower, "two_tower"), _summarize(ranks_scoring, "scoring")


@torch.inference_mode()
def evaluate_centroid_baseline(
    val_pairs: list[TrainPair],
    matrix: np.ndarray,
    *,
    device: torch.device,
    n_samples: int = 2000,
    n_negatives: int = 100,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    weight_small: float = 0.5,
    weight_medium: float = 0.2,
    weight_large: float = 0.3,
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    pool_t = torch.from_numpy(matrix).to(device)
    n_pool = matrix.shape[0]
    if len(val_pairs) > n_samples:
        idxs = rng.choice(len(val_pairs), size=n_samples, replace=False)
    else:
        idxs = np.arange(len(val_pairs))
    ranks: list[int] = []
    for i in idxs:
        p = val_pairs[i]
        q = _baseline_query(p, w_small=weight_small, w_med=weight_medium, w_lg=weight_large)
        q_t = torch.from_numpy(q.astype(np.float32)).to(device)
        target_row = int(np.argmax(matrix @ p.target))
        neg_idx = rng.choice(n_pool, size=n_negatives + 5, replace=False)
        neg_idx = neg_idx[neg_idx != target_row][:n_negatives]
        cand_rows = np.concatenate([[target_row], neg_idx])
        cand_emb = pool_t[cand_rows]
        scores = cand_emb @ q_t
        order = torch.argsort(scores, descending=True)
        target_pos = int((order == 0).nonzero(as_tuple=False).item())
        ranks.append(target_pos + 1)
    arr = np.asarray(ranks, dtype=np.int64)
    out = {
        "n_samples": int(len(arr)),
        "mrr": float(np.mean(1.0 / np.maximum(arr, 1))),
        "median_rank": float(np.median(arr)),
        "mode": f"baseline-centroid-{n_negatives}neg",
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(arr <= k))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@click.command()
@click.option("--store", "store_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--events", "events_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--temperature", type=float, default=0.07, show_default=True)
@click.option("--cosine-weight", type=float, default=0.5, show_default=True)
@click.option(
    "--scoring-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight of the scoring-head listwise-CE loss vs the two-tower InfoNCE.",
)
@click.option(
    "--n-train-negatives",
    type=int,
    default=64,
    show_default=True,
    help="Negatives per training row for the listwise scoring loss.",
)
@click.option(
    "--hard-negative-frac",
    type=float,
    default=0.5,
    show_default=True,
    help="After warmup, fraction of negatives drawn from top-K of the "
    "two-tower head (hard negatives) rather than random. Set to 0 to disable.",
)
@click.option(
    "--hard-negative-warmup-epochs",
    type=int,
    default=3,
    show_default=True,
    help="First N epochs use 100 %% random negatives so the scorer doesn't "
    "collapse before the state encoder converges.",
)
@click.option("--val-frac-users", type=float, default=0.15, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    store_path: Path,
    events_path: Path,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    cosine_weight: float,
    scoring_weight: float,
    n_train_negatives: int,
    hard_negative_frac: float,
    hard_negative_warmup_epochs: int,
    val_frac_users: float,
    device: str,
    seed: int,
) -> None:
    from student.device import resolve_torch_device

    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    store = EmbeddingStore.open(store_path)
    matrix = store._matrix
    if matrix is None:
        store.rebuild_matrix()
        matrix = store._matrix
    assert matrix is not None
    track_id_to_row = {tid: i for i, tid in enumerate(store.df["track_id"].astype(str).tolist())}

    events = pd.read_parquet(events_path)
    events = events[events["track_id"].astype(str).isin(track_id_to_row)].reset_index(drop=True)
    print(f"events: {len(events)} rows, users: {events['user_id'].nunique()}")

    rng = np.random.default_rng(seed)
    users = sorted(events["user_id"].unique())
    rng.shuffle(users)
    n_val_users = max(1, int(round(len(users) * val_frac_users)))
    val_users = set(users[:n_val_users])
    train_users = set(users[n_val_users:])
    print(f"users: train={len(train_users)} val={len(val_users)}")

    train_pairs: list[TrainPair] = []
    val_pairs: list[TrainPair] = []
    for user_id, group in tqdm(events.groupby("user_id"), desc="featurize", unit="user"):
        pairs = _build_pairs_for_user(group, matrix, track_id_to_row)
        if user_id in val_users:
            val_pairs.extend(pairs)
        else:
            train_pairs.extend(pairs)

    def _has_session_history(p: TrainPair) -> bool:
        diffs = np.abs(p.history_small[1:] - p.history_small[:-1]).sum()
        return float(diffs) > 1e-6

    train_pairs = [p for p in train_pairs if _has_session_history(p)]
    val_pairs = [p for p in val_pairs if _has_session_history(p)]
    print(f"after history>=2 filter: train={len(train_pairs)} val={len(val_pairs)}")

    base = evaluate_centroid_baseline(val_pairs, matrix, device=torch_device, rng_seed=seed)
    print(
        f"baseline[100-neg]: recall@1={base['recall@1']:.4f} "
        f"recall@10={base['recall@10']:.4f} mrr={base['mrr']:.4f}"
    )

    state_cfg = HistoryPredictorConfig(embedding_dim=int(matrix.shape[1]), context_k=CONTEXT_K)
    score_cfg = ScoringHeadConfig(state_dim=int(matrix.shape[1]), candidate_dim=int(matrix.shape[1]))
    model = HistoryAwareWithScorer(state_cfg, score_cfg).to(torch_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"predictor (state+scorer): {n_params:,} parameters")

    train_ds = HistoryEventDataset(train_pairs)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    n_steps = max(1, len(train_loader)) * epochs

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.05, anneal_strategy="cos",
    )

    pool_t = torch.from_numpy(matrix).to(torch_device)
    n_pool = matrix.shape[0]

    best_recall_at_1 = -1.0
    best_report: dict[str, Any] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng_train = np.random.default_rng(seed + 1)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nce = 0.0
        epoch_score_ce = 0.0
        epoch_pos_cos = 0.0
        n_batches = 0

        for h_small, h_med, h_lg, tf, sf, tgt in train_loader:
            h_small = h_small.to(torch_device)
            h_med = h_med.to(torch_device)
            h_lg = h_lg.to(torch_device)
            tf = tf.to(torch_device)
            sf = sf.to(torch_device)
            tgt = tgt.to(torch_device)
            tgt = F.normalize(tgt, p=2.0, dim=-1)

            state = model.encode_state(h_small, h_med, h_lg, tf, sf)        # (B, D)

            # Two-tower InfoNCE on cosine.
            nce = info_nce(state, tgt, temperature=temperature)
            cos_pos = (state * tgt).sum(dim=-1).mean()

            # Listwise scoring head: 1 positive + N negatives per row.
            # Composition: random + (after warmup) top-K from the two-tower
            # head as hard negatives. Hard negatives force the scorer to
            # discriminate between "almost right" and "right" rather than
            # "easy random track" vs "right".
            B = state.size(0)
            use_hard = epoch > hard_negative_warmup_epochs and hard_negative_frac > 0
            n_hard = int(round(n_train_negatives * hard_negative_frac)) if use_hard else 0
            n_random = n_train_negatives - n_hard

            neg_emb_parts: list[torch.Tensor] = []
            if n_random > 0:
                rand_rows = rng_train.integers(0, n_pool, size=(B, n_random))
                neg_emb_parts.append(pool_t[rand_rows.flatten()].view(B, n_random, -1))
            if n_hard > 0:
                # Cosine scores of state vs full library, mask out "exact target"
                # so we never sample the positive as a hard negative.
                with torch.no_grad():
                    sim = state @ pool_t.T                               # (B, N_pool)
                    # Find each row's target index by argmax against tgt;
                    # fall back to similarity to tgt as identifier.
                    target_sim = (state * tgt).sum(dim=-1, keepdim=True)  # (B, 1)
                    # Mark target-row(s) — approx by setting top-1 row to -inf if
                    # its embedding equals target. Equivalent: subtract a peak
                    # at the closest-to-tgt row.
                    tgt_sim_all = pool_t @ tgt.T                           # (N_pool, B)
                    tgt_rows = tgt_sim_all.argmax(dim=0)                   # (B,)
                    sim.scatter_(1, tgt_rows.unsqueeze(1), float("-inf"))
                    # top-k over a wider pool than we need so sampling has variety
                    pool_size = max(n_hard * 4, 50)
                    _, top_idx = sim.topk(pool_size, dim=1)                # (B, pool_size)
                    # uniformly sample n_hard from the pool per row
                    rand_pick = torch.randint(
                        0, pool_size, (B, n_hard), device=torch_device
                    )
                    hard_rows = top_idx.gather(1, rand_pick)               # (B, n_hard)
                    hard_emb = pool_t[hard_rows.flatten()].view(B, n_hard, -1)
                neg_emb_parts.append(hard_emb)

            neg_emb = torch.cat(neg_emb_parts, dim=1)
            cand_emb = torch.cat([tgt.unsqueeze(1), neg_emb], dim=1)         # (B, 1+N, D)
            scores = model.score(state, cand_emb)                             # (B, 1+N)
            labels = torch.zeros(B, dtype=torch.long, device=torch_device)    # positive at index 0
            score_ce = F.cross_entropy(scores, labels)

            loss = nce + cosine_weight * (1.0 - cos_pos) + scoring_weight * score_ce

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            scheduler.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_nce += float(nce.detach().cpu())
            epoch_score_ce += float(score_ce.detach().cpu())
            epoch_pos_cos += float(cos_pos.detach().cpu())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_nce = epoch_nce / max(1, n_batches)
        avg_score_ce = epoch_score_ce / max(1, n_batches)
        avg_cos = epoch_pos_cos / max(1, n_batches)

        val_tt, val_sc = evaluate_sampled_both(
            model, val_pairs, matrix, device=torch_device, rng_seed=seed + epoch
        )
        line = (
            f"epoch={epoch:02d}/{epochs} loss={avg_loss:.4f} nce={avg_nce:.4f} "
            f"score_ce={avg_score_ce:.4f} train_cos={avg_cos:.4f} "
            f"val_2tower[r@1={val_tt['recall@1']:.4f} r@10={val_tt['recall@10']:.4f} mrr={val_tt['mrr']:.4f}] "
            f"val_score[r@1={val_sc['recall@1']:.4f} r@10={val_sc['recall@10']:.4f} mrr={val_sc['mrr']:.4f}]"
        )
        print(line, flush=True)

        # Track best on the scoring-head's recall@1 (the metric we're optimizing for).
        if val_sc["recall@1"] > best_recall_at_1:
            best_recall_at_1 = val_sc["recall@1"]
            best_report = {
                "epoch": epoch,
                "val_two_tower": val_tt,
                "val_scoring": val_sc,
            }
            torch.save(
                {
                    "predictor_version": SCORING_PREDICTOR_VERSION,
                    "model_state_dict": model.state_dict(),
                    "state_config": state_cfg.to_dict(),
                    "scoring_config": score_cfg.to_dict(),
                    "store_path": str(store_path),
                    "events_path": str(events_path),
                    "best_report": best_report,
                },
                out_path,
            )

    final_report = {
        "predictor_version": SCORING_PREDICTOR_VERSION,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_train_users": len(train_users),
        "n_val_users": len(val_users),
        "n_params": n_params,
        "best": best_report,
        "baseline_centroid_100neg": base,
        "state_config": state_cfg.to_dict(),
        "scoring_config": score_cfg.to_dict(),
    }
    sys.stdout.write(json.dumps(final_report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
