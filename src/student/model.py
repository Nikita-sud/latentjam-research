"""A compact 2D CNN student for CLAP-space audio embeddings."""

from __future__ import annotations

import torch
from torch import nn

from student.config import STUDENT_PARAM_MAX, STUDENT_PARAM_MIN, TEACHER_EMBEDDING_DIM


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ConvBnAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
    ):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int | tuple[int, int] = 1,
    ):
        super().__init__()
        self.depthwise = ConvBnAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = ConvBnAct(in_channels, out_channels, kernel_size=1)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise(self.depthwise(x))
        if self.use_residual:
            return x + out
        return out


class MelCnnStudent(nn.Module):
    """Mobile-friendly CNN that maps ``(B, 1, 96, T)`` log-mels to 512-d CLAP space."""

    def __init__(
        self,
        embedding_dim: int = TEACHER_EMBEDDING_DIM,
        dropout: float = 0.1,
        enforce_param_budget: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.features = nn.Sequential(
            ConvBnAct(1, 128, stride=(2, 2)),
            DepthwiseSeparableBlock(128, 192, stride=(1, 2)),
            DepthwiseSeparableBlock(192, 256, stride=(2, 2)),
            DepthwiseSeparableBlock(256, 256),
            DepthwiseSeparableBlock(256, 384, stride=(2, 2)),
            DepthwiseSeparableBlock(384, 384),
            DepthwiseSeparableBlock(384, 512, stride=(2, 2)),
            DepthwiseSeparableBlock(512, 512),
            DepthwiseSeparableBlock(512, 768, stride=(2, 2)),
            DepthwiseSeparableBlock(768, 768),
            DepthwiseSeparableBlock(768, 1024, stride=(1, 2)),
            DepthwiseSeparableBlock(1024, 1024),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, embedding_dim),
        )

        if enforce_param_budget:
            n_params = count_parameters(self)
            if not STUDENT_PARAM_MIN <= n_params <= STUDENT_PARAM_MAX:
                raise ValueError(
                    f"student has {n_params:,} parameters, expected "
                    f"{STUDENT_PARAM_MIN:,}..{STUDENT_PARAM_MAX:,}"
                )

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim != 4:
            raise ValueError(f"expected log_mel shape (B, 1, M, T), got {tuple(log_mel.shape)}")
        x = self.features(log_mel)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return nn.functional.normalize(x, p=2.0, dim=-1)

    def config_dict(self) -> dict[str, int | float | str]:
        return {
            "architecture": "MelCnnStudent",
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "parameters": count_parameters(self),
        }
