"""Projector head used for self-supervised training.

Drops at inference; only used to map backbone features into the SSL loss space.
Three Linear layers with BN+GELU between, no activation on the output.
"""

from __future__ import annotations

import torch
from torch import nn


class VICRegProjector(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 2048, out_dim: int = 2048):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config_dict(self) -> dict:
        return {
            "architecture": "VICRegProjector",
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
