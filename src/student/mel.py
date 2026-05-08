"""Log-mel frontend used to train the student model."""

from __future__ import annotations

import torch
from torch import nn

from student.config import MelConfig


def ensure_window_length(waveform: torch.Tensor, window_samples: int) -> torch.Tensor:
    """Pad or trim a waveform tensor to ``window_samples`` on the last axis."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"expected waveform shape (B, T) or (T,), got {tuple(waveform.shape)}")

    n = waveform.shape[-1]
    if n == window_samples:
        return waveform
    if n > window_samples:
        return waveform[..., :window_samples]
    return nn.functional.pad(waveform, (0, window_samples - n))


class LogMelExtractor(nn.Module):
    """Convert fixed-length waveform windows into normalized log-mel tensors."""

    def __init__(self, config: MelConfig | None = None):
        super().__init__()
        self.config = config or MelConfig()

        # Local import keeps lightweight tests/imports from paying torchaudio startup cost.
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            center=True,
            normalized=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = ensure_window_length(waveform, self.config.window_samples)
        mel = self.mel(waveform)
        log_mel = torch.log(torch.clamp(mel, min=self.config.amin))
        if self.config.normalize:
            mean = log_mel.mean(dim=(-2, -1), keepdim=True)
            std = log_mel.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
            log_mel = (log_mel - mean) / std
        return log_mel.unsqueeze(1)
