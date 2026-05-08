"""Torch device selection helpers for student distillation CLIs."""

from __future__ import annotations

import torch


def resolve_torch_device(requested: str) -> torch.device:
    """Resolve ``auto`` and validate explicit accelerator requests."""
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is false. "
            "Run the command outside the sandbox or fix the PyTorch/Mac setup."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false."
        )
    return device
