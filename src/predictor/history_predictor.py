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
