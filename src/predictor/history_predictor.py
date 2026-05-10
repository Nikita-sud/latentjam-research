"""History-aware session predictor.

Inputs the predictor needs at inference time (matches what the Android
app's ``ListeningEventRecorder`` will provide):

  - history_small (4, D)  : stack of last K=4 played-track embeddings.
                            Includes positional encoding so the order
                            within the small window matters.
  - history_medium  (D,)  : recency-weighted (30-day exp decay) centroid
                            of completed plays.
  - history_large   (D,)  : long-term recency+frequency-weighted centroid
                            (top-N played, log-decay by recency).
  - time_features  (5,)   : sin(hour*2pi/24), cos(...), sin(dow*2pi/7),
                            cos(...), is_weekend.
  - session_features (4,) : log(session_pos+1), was_last_skipped,
                            log(time_since_session_start_min+1),
                            recent_completion_rate.

Output:
  predicted_next_track_emb (D,) -- L2 normalized; the app does cosine
  retrieval over the EmbeddingStore matrix to fetch top-K candidates,
  excluding the seeds.

Architecture:
  - Small Transformer encoder over history_small (with learned position
    embeddings) -> mean-pool -> 512-d.
  - history_medium and history_large are already centroids; pass through
    a tiny MLP each to project into the fusion space.
  - time_features and session_features get a small MLP -> 64-d.
  - Concat all -> fusion MLP -> residual to history_small_centroid ->
    L2 normalize -> output.

Param budget ~3 M, CPU forward ~50 ms — fits on-device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


HISTORY_PREDICTOR_VERSION = "history-mlp@k4-d512+time5+session4/v1"
SCORING_PREDICTOR_VERSION = "history-mlp+scoring@k4-d512+time5+session4/v1"


@dataclass
class HistoryPredictorConfig:
    embedding_dim: int = 512
    context_k: int = 4
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff: int = 1024
    fuse_hidden: int = 1024
    time_feat_dim: int = 5
    session_feat_dim: int = 4
    aux_hidden: int = 64
    dropout: float = 0.1
    residual_scale: float = 0.4

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedding_dim": self.embedding_dim,
            "context_k": self.context_k,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "transformer_ff": self.transformer_ff,
            "fuse_hidden": self.fuse_hidden,
            "time_feat_dim": self.time_feat_dim,
            "session_feat_dim": self.session_feat_dim,
            "aux_hidden": self.aux_hidden,
            "dropout": self.dropout,
            "residual_scale": self.residual_scale,
            "predictor_version": HISTORY_PREDICTOR_VERSION,
        }


class HistoryAwarePredictor(nn.Module):
    def __init__(self, cfg: HistoryPredictorConfig):
        super().__init__()
        self.cfg = cfg

        d = cfg.embedding_dim

        # Position embeddings for the K-track small history.
        self.position = nn.Embedding(cfg.context_k, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.h_small_enc = nn.TransformerEncoder(encoder_layer, num_layers=cfg.transformer_layers)

        # Tiny projectors for the centroid histories so they compete fairly
        # with the transformer-encoded small history.
        self.h_medium_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.h_large_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.time_enc = nn.Sequential(
            nn.Linear(cfg.time_feat_dim, cfg.aux_hidden),
            nn.GELU(),
            nn.Linear(cfg.aux_hidden, cfg.aux_hidden),
        )
        self.session_enc = nn.Sequential(
            nn.Linear(cfg.session_feat_dim, cfg.aux_hidden),
            nn.GELU(),
            nn.Linear(cfg.aux_hidden, cfg.aux_hidden),
        )

        fuse_in = 3 * d + 2 * cfg.aux_hidden
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, cfg.fuse_hidden),
            nn.LayerNorm(cfg.fuse_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fuse_hidden, cfg.fuse_hidden),
            nn.LayerNorm(cfg.fuse_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fuse_hidden, d),
        )

        self.residual_scale = nn.Parameter(torch.tensor(cfg.residual_scale))

    def forward(
        self,
        history_small: torch.Tensor,   # (B, K, D)
        history_medium: torch.Tensor,  # (B, D)
        history_large: torch.Tensor,   # (B, D)
        time_features: torch.Tensor,   # (B, time_feat_dim)
        session_features: torch.Tensor,  # (B, session_feat_dim)
    ) -> torch.Tensor:
        b, k, d = history_small.shape
        assert k == self.cfg.context_k and d == self.cfg.embedding_dim

        positions = torch.arange(k, device=history_small.device)
        pos_emb = self.position(positions)[None, :, :]
        h_small = history_small + pos_emb
        h_small = self.h_small_enc(h_small)         # (B, K, D)
        h_small_pool = h_small.mean(dim=1)          # (B, D)

        h_med = self.h_medium_proj(history_medium)  # (B, D)
        h_lg = self.h_large_proj(history_large)     # (B, D)
        t = self.time_enc(time_features)            # (B, aux_hidden)
        s = self.session_enc(session_features)      # (B, aux_hidden)

        x = torch.cat([h_small_pool, h_med, h_lg, t, s], dim=-1)
        offset = self.fuse(x)                       # (B, D)

        # Anchor on history_small_pool so cold-start mediums/larges can't
        # corrupt the prediction; the fusion MLP only adds a residual.
        out = h_small_pool + self.residual_scale * offset
        return F.normalize(out, p=2.0, dim=-1)


# ---------------------------------------------------------------------------
# Scoring head (cross-encoder reranker)
# ---------------------------------------------------------------------------


@dataclass
class ScoringHeadConfig:
    state_dim: int = 512
    candidate_dim: int = 512
    hidden: int = 512
    dropout: float = 0.1
    use_state_anchor: bool = True  # add cosine(state, candidate) as a raw input

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "candidate_dim": self.candidate_dim,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "use_state_anchor": self.use_state_anchor,
            "predictor_version": SCORING_PREDICTOR_VERSION,
        }


class ScoringHead(nn.Module):
    """Cross-encoder: ``score(state, candidate) -> scalar``.

    Inputs the four standard cross-encoder features:
      - state                  (D,)
      - candidate              (D,)
      - state * candidate      (D,) elementwise product (encodes alignment)
      - |state - candidate|    (D,) elementwise abs diff (encodes distance)
      - cosine(state, candidate)  scalar (raw alignment, optional)

    Body: small MLP -> scalar logit. Trained with listwise softmax CE over
    (1 positive + N negatives), so it directly optimizes top-1 ranking.
    """

    def __init__(self, cfg: ScoringHeadConfig):
        super().__init__()
        self.cfg = cfg
        feat_dim = 4 * cfg.candidate_dim + (1 if cfg.use_state_anchor else 0)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Score every candidate against the state.

        state:      (B, D) or (D,)
        candidates: (B, N, D) or (N, D) — broadcast B over candidates if state has B.

        Returns: (B, N) or (N,) scalar scores.
        """
        squeeze = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True
        if candidates.dim() == 2:
            # (N, D): use the same N for the single state.
            candidates = candidates.unsqueeze(0)
        b, n, d = candidates.shape
        if state.size(0) == 1 and b > 1:
            state = state.expand(b, -1)
        # broadcast state across N candidates -> (B, N, D)
        s = state.unsqueeze(1).expand(-1, n, -1)
        prod = s * candidates
        diff = (s - candidates).abs()
        feats = [s, candidates, prod, diff]
        if self.cfg.use_state_anchor:
            cos = (s * candidates).sum(dim=-1, keepdim=True)
            feats.append(cos)
        x = torch.cat(feats, dim=-1)             # (B, N, 4D + ?)
        scores = self.mlp(x).squeeze(-1)          # (B, N)
        if squeeze:
            scores = scores.squeeze(0)
        return scores


class HistoryAwareWithScorer(nn.Module):
    """HistoryAwarePredictor (state encoder) + ScoringHead (cross-encoder).

    At inference the app can do either:
    - **two-tower**:  state = state_encoder(...); cosine vs library matrix.
    - **scorer**:     score = scoring_head(state, candidate); pick top-k.

    Train both heads jointly so the state encoder produces a vector that's
    useful for both fast retrieval (cosine) and precise scoring.
    """

    def __init__(self, state_cfg: HistoryPredictorConfig, score_cfg: ScoringHeadConfig):
        super().__init__()
        self.state_encoder = HistoryAwarePredictor(state_cfg)
        self.scoring_head = ScoringHead(score_cfg)

    def encode_state(
        self,
        history_small: torch.Tensor,
        history_medium: torch.Tensor,
        history_large: torch.Tensor,
        time_features: torch.Tensor,
        session_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_encoder(
            history_small, history_medium, history_large, time_features, session_features
        )

    def score(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.scoring_head(state, candidates)
