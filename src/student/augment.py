"""Augmentations for self-supervised training of the mel-CNN student.

Two waveform-level ops (`random_gain`, `random_polarity`) and one mel-level op
(`SpecAugment`). All are MPS-safe (no per-element loops in inner kernels) and
return new tensors so autograd is unaffected.
"""

from __future__ import annotations

import torch
from torch import nn


def random_gain(waveform: torch.Tensor, max_db: float = 3.0) -> torch.Tensor:
    """Apply per-sample gain in ``[-max_db, +max_db]`` dB.

    ``waveform`` is shape ``(B, S)``; one gain value per batch element.
    """
    if max_db <= 0:
        return waveform
    db = (torch.rand(waveform.shape[0], device=waveform.device, dtype=waveform.dtype) * 2.0 - 1.0) * max_db
    gain = torch.pow(torch.tensor(10.0, device=waveform.device, dtype=waveform.dtype), db / 20.0)
    return waveform * gain.unsqueeze(-1)


def random_polarity(waveform: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Flip polarity (multiply by -1) with probability ``p`` per batch element.

    Cosine-similar audio is invariant to polarity, so this is a free augmentation.
    """
    if p <= 0:
        return waveform
    flip = (torch.rand(waveform.shape[0], device=waveform.device) < p).to(waveform.dtype) * -2.0 + 1.0
    return waveform * flip.unsqueeze(-1)


class SpecAugment(nn.Module):
    """Standard SpecAugment: ``freq_masks`` freq bands and ``time_masks`` time bands.

    Each call samples fresh masks. The same masks are applied across the batch
    (paper-faithful and faster on MPS than per-element masking). No-op in eval.
    """

    def __init__(
        self,
        freq_masks: int = 2,
        freq_mask_max: int = 8,
        time_masks: int = 2,
        time_mask_max: int = 25,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_mask_max = freq_mask_max
        self.time_masks = time_masks
        self.time_mask_max = time_mask_max

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.training or (self.freq_masks <= 0 and self.time_masks <= 0):
            return mel
        if mel.ndim != 4:
            raise ValueError(f"expected (B, C, M, T), got {tuple(mel.shape)}")
        out = mel.clone()
        _b, _c, M, T = out.shape

        for _ in range(self.freq_masks):
            f = int(torch.randint(1, self.freq_mask_max + 1, (1,)).item())
            f = min(f, max(1, M - 1))
            f0 = int(torch.randint(0, max(1, M - f), (1,)).item())
            out[:, :, f0 : f0 + f, :] = 0

        for _ in range(self.time_masks):
            t = int(torch.randint(1, self.time_mask_max + 1, (1,)).item())
            t = min(t, max(1, T - 1))
            t0 = int(torch.randint(0, max(1, T - t), (1,)).item())
            out[:, :, :, t0 : t0 + t] = 0

        return out
