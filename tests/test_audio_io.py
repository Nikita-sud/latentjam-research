from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from encoder.audio_io import TARGET_SR, AudioLoadError, load_audio, windows


def test_load_audio_returns_canonical_shape(tmp_audio_wav: Path) -> None:
    wav = load_audio(tmp_audio_wav)
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    assert wav.shape[0] == TARGET_SR  # 1 second @ 24 kHz
    assert wav.min() >= -1.0
    assert wav.max() <= 1.0


def test_load_audio_resamples_when_sr_mismatches(tmp_path: Path) -> None:
    import soundfile as sf

    src_sr = 44_100
    duration_s = 0.5
    t = np.arange(int(src_sr * duration_s), dtype=np.float32) / src_sr
    wav = 0.25 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)

    out = tmp_path / "src_44k.wav"
    sf.write(out, wav, src_sr, subtype="FLOAT")

    loaded = load_audio(out)
    expected = int(round(duration_s * TARGET_SR))
    assert abs(loaded.shape[0] - expected) <= 1


def test_load_audio_downmixes_stereo(tmp_path: Path) -> None:
    import soundfile as sf

    sr = TARGET_SR
    duration_s = 0.25
    t = np.arange(int(sr * duration_s), dtype=np.float32) / sr
    left = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    right = 0.5 * np.sin(2.0 * np.pi * 880.0 * t).astype(np.float32)
    stereo = np.stack([left, right], axis=1)

    out = tmp_path / "stereo.wav"
    sf.write(out, stereo, sr, subtype="FLOAT")

    mono = load_audio(out)
    assert mono.ndim == 1
    np.testing.assert_allclose(mono, (left + right) / 2.0, atol=1e-6)


def test_load_audio_raises_for_garbage(tmp_path: Path) -> None:
    bad = tmp_path / "not_audio.flac"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioLoadError):
        load_audio(bad)


def test_windows_short_track_zero_pads() -> None:
    wav = np.ones(1000, dtype=np.float32)
    win = windows(wav, window_samples=120_000, hop_samples=60_000)
    assert win.shape == (1, 120_000)
    assert (win[0, :1000] == 1.0).all()
    assert (win[0, 1000:] == 0.0).all()


def test_windows_long_track_overlaps_correctly() -> None:
    wav = np.arange(300_000, dtype=np.float32)
    win = windows(wav, window_samples=120_000, hop_samples=60_000)
    # Expected windows starting at 0, 60_000, 120_000, 180_000.
    assert win.shape[0] == 4
    assert win.shape[1] == 120_000
    np.testing.assert_array_equal(win[0, 0:5], np.array([0, 1, 2, 3, 4], dtype=np.float32))
    np.testing.assert_array_equal(
        win[1, 0:3], np.array([60_000, 60_001, 60_002], dtype=np.float32)
    )


def test_windows_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        windows(np.zeros((2, 10), dtype=np.float32), window_samples=4, hop_samples=2)
    with pytest.raises(ValueError):
        windows(np.zeros(10, dtype=np.float32), window_samples=0, hop_samples=2)
