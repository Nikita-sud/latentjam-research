"""Audio file loading + resampling to the canonical 24 kHz mono float32 format.

The canonical shape is fixed: see ``CLAUDE.md`` for why. Upstream code (the MERT
encoder, the ONNX export wrapper, the on-device runtime) all assume this exact
sample rate and channel layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf

TARGET_SR: Final[int] = 24_000

SUPPORTED_EXTS: Final[frozenset[str]] = frozenset(
    {".flac", ".wav", ".ogg", ".opus", ".mp3", ".m4a"}
)


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be decoded into the canonical format."""


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file as a mono float32 waveform at ``target_sr``.

    Returns a 1-D ``np.ndarray`` of dtype ``float32`` with values in ``[-1, 1]``.
    Stereo or multi-channel input is downmixed by mean. Sample rate mismatches
    are resampled via ``resampy`` (band-limited sinc, kaiser-best by default).
    """
    p = Path(path)
    try:
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as e:
        raise AudioLoadError(f"Failed to decode {p}: {e}") from e

    if wav.ndim == 2:
        wav = wav.mean(axis=1, dtype=np.float32)
    elif wav.ndim != 1:
        raise AudioLoadError(f"Unexpected waveform shape {wav.shape} from {p}")

    wav = wav.astype(np.float32, copy=False)

    if sr != target_sr:
        # Local import keeps `import audio_io` cheap; resampy pulls numba/llvmlite.
        import resampy

        wav = resampy.resample(wav, sr, target_sr, filter="kaiser_best").astype(
            np.float32, copy=False
        )

    return wav


def windows(
    waveform: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Split a 1-D waveform into ``(n_windows, window_samples)`` chunks.

    The last incomplete chunk is zero-padded so that very short tracks still
    produce one window. Shapes are deterministic given the inputs, which the
    ONNX export path depends on.
    """
    if waveform.ndim != 1:
        raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive")

    n = waveform.shape[0]
    if n <= window_samples:
        out = np.zeros((1, window_samples), dtype=np.float32)
        out[0, :n] = waveform
        return out

    n_windows = 1 + (n - window_samples + hop_samples - 1) // hop_samples
    out = np.zeros((n_windows, window_samples), dtype=np.float32)
    for i in range(n_windows):
        start = i * hop_samples
        end = min(start + window_samples, n)
        out[i, : end - start] = waveform[start:end]
    return out
