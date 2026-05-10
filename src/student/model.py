"""Compact student encoders for CLAP-space audio embeddings."""

from __future__ import annotations

import torch
from torch import nn

from student.config import (
    MelConfig,
    STUDENT_PARAM_MAX,
    STUDENT_PARAM_MIN,
    TEACHER_EMBEDDING_DIM,
)

STUDENT_ARCH_CHOICES = ("mel-cnn", "passt-s")


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


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _grid_size(
    input_size: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, int]:
    freq = 1 + (input_size[0] - patch_size[0]) // stride[0]
    time = 1 + (input_size[1] - patch_size[1]) // stride[1]
    if freq <= 0 or time <= 0:
        raise ValueError(
            f"invalid patch grid for input={input_size}, patch={patch_size}, stride={stride}"
        )
    return freq, time


class PaSSTSmallStudent(nn.Module):
    """Small PaSST-style spectrogram transformer for 96-bin log-mels.

    This is intentionally sized near the existing mobile CNN budget: patch
    embedding + 4 transformer blocks + a small projection head lands around
    7M trainable parameters while keeping the same 512-d normalized output.
    Structured patchout drops complete frequency/time patch positions during
    training only; eval uses the full spectrogram grid.
    """

    def __init__(
        self,
        embedding_dim: int = TEACHER_EMBEDDING_DIM,
        input_size: tuple[int, int] | None = None,
        patch_size: int | tuple[int, int] = (16, 16),
        stride: int | tuple[int, int] = (16, 16),
        embed_dim: int = 384,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 3.0,
        head_hidden_dim: int = 1024,
        dropout: float = 0.1,
        patchout_freq: int = 1,
        patchout_time: int = 3,
        enforce_param_budget: bool = True,
    ):
        super().__init__()
        mel_cfg = MelConfig()
        self.embedding_dim = int(embedding_dim)
        self.input_size = input_size or (
            mel_cfg.n_mels,
            1 + mel_cfg.window_samples // mel_cfg.hop_length,
        )
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.head_hidden_dim = int(head_hidden_dim)
        self.dropout = float(dropout)
        self.patchout_freq = int(patchout_freq)
        self.patchout_time = int(patchout_time)

        grid = _grid_size(self.input_size, self.patch_size, self.stride)
        self.grid_size = grid
        self.patch_embed = nn.Conv2d(
            1,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride,
            bias=True,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.freq_pos = nn.Parameter(torch.zeros(1, self.embed_dim, grid[0], 1))
        self.time_pos = nn.Parameter(torch.zeros(1, self.embed_dim, 1, grid[1]))
        self.pos_drop = nn.Dropout(self.dropout)

        ff_dim = int(round(self.embed_dim * self.mlp_ratio))
        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=self.depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden_dim, self.embedding_dim),
        )
        self._init_weights()

        if enforce_param_budget:
            n_params = count_parameters(self)
            if not STUDENT_PARAM_MIN <= n_params <= STUDENT_PARAM_MAX:
                raise ValueError(
                    f"student has {n_params:,} parameters, expected "
                    f"{STUDENT_PARAM_MIN:,}..{STUDENT_PARAM_MAX:,}"
                )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.freq_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _drop_patch_positions(x: torch.Tensor, dim: int, n_drop: int) -> torch.Tensor:
        if n_drop <= 0:
            return x
        size = x.shape[dim]
        keep = max(1, size - min(int(n_drop), size - 1))
        if keep == size:
            return x
        idx = torch.randperm(size, device=x.device)[:keep].sort().values
        return x.index_select(dim, idx)

    def forward(self, log_mel: torch.Tensor) -> torch.Tensor:
        if log_mel.ndim != 4:
            raise ValueError(f"expected log_mel shape (B, 1, M, T), got {tuple(log_mel.shape)}")
        x = self.patch_embed(log_mel)
        _b, _c, freq, time = x.shape
        if freq > self.freq_pos.shape[2] or time > self.time_pos.shape[3]:
            raise ValueError(
                f"patch grid {(freq, time)} exceeds configured grid {self.grid_size}; "
                "construct PaSSTSmallStudent with a larger input_size"
            )
        x = x + self.freq_pos[:, :, :freq, :] + self.time_pos[:, :, :, :time]
        if self.training:
            x = self._drop_patch_positions(x, dim=2, n_drop=self.patchout_freq)
            x = self._drop_patch_positions(x, dim=3, n_drop=self.patchout_time)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return nn.functional.normalize(x, p=2.0, dim=-1)

    def config_dict(self) -> dict[str, int | float | str | tuple[int, int]]:
        return {
            "architecture": "PaSSTSmallStudent",
            "embedding_dim": self.embedding_dim,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "head_hidden_dim": self.head_hidden_dim,
            "dropout": self.dropout,
            "patchout_freq": self.patchout_freq,
            "patchout_time": self.patchout_time,
            "parameters": count_parameters(self),
        }


def _normalize_architecture(architecture: str) -> str:
    arch = architecture.strip().lower().replace("_", "-")
    if arch in {"mel-cnn", "melcnn", "melcnnstudent"}:
        return "mel-cnn"
    if arch in {"passt-s", "passt", "passtsmallstudent", "passt-small"}:
        return "passt-s"
    raise ValueError(
        f"unknown student architecture {architecture!r}; expected one of {STUDENT_ARCH_CHOICES}"
    )


def build_student_model(
    architecture: str,
    *,
    embedding_dim: int = TEACHER_EMBEDDING_DIM,
    passt_patchout_freq: int = 1,
    passt_patchout_time: int = 3,
) -> nn.Module:
    arch = _normalize_architecture(architecture)
    if arch == "mel-cnn":
        return MelCnnStudent(embedding_dim=embedding_dim)
    return PaSSTSmallStudent(
        embedding_dim=embedding_dim,
        patchout_freq=passt_patchout_freq,
        patchout_time=passt_patchout_time,
    )


def build_student_from_config(config: dict) -> nn.Module:
    architecture = str(config.get("architecture", "MelCnnStudent"))
    arch = _normalize_architecture(architecture)
    embedding_dim = int(config.get("embedding_dim", TEACHER_EMBEDDING_DIM))
    if arch == "mel-cnn":
        return MelCnnStudent(
            embedding_dim=embedding_dim,
            dropout=float(config.get("dropout", 0.1)),
        )

    return PaSSTSmallStudent(
        embedding_dim=embedding_dim,
        input_size=tuple(config.get("input_size", (96, 251))),  # type: ignore[arg-type]
        patch_size=tuple(config.get("patch_size", (16, 16))),  # type: ignore[arg-type]
        stride=tuple(config.get("stride", (16, 16))),  # type: ignore[arg-type]
        embed_dim=int(config.get("embed_dim", 384)),
        depth=int(config.get("depth", 4)),
        num_heads=int(config.get("num_heads", 6)),
        mlp_ratio=float(config.get("mlp_ratio", 3.0)),
        head_hidden_dim=int(config.get("head_hidden_dim", 1024)),
        dropout=float(config.get("dropout", 0.1)),
        patchout_freq=int(config.get("patchout_freq", 1)),
        patchout_time=int(config.get("patchout_time", 3)),
    )
