from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_audio_wav(tmp_path: Path) -> Path:
    """A 1-second 24 kHz mono sine wave at 440 Hz, written as float32 WAV.

    Used to exercise audio loaders without depending on real music files.
    """
    import soundfile as sf

    sr = 24000
    duration_s = 1.0
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    wav = 0.25 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    out = tmp_path / "tone_440.wav"
    sf.write(out, wav, sr, subtype="FLOAT")
    return out
