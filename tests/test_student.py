from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from student.config import STUDENT_PARAM_MAX, STUDENT_PARAM_MIN, WINDOW_SAMPLES, MelConfig
from student.data import DistillTargetDataset, window_starts
from student.device import resolve_torch_device
from student.mel import LogMelExtractor, ensure_window_length
from student.metrics import topk_overlap
from student.model import (
    MelCnnStudent,
    PaSSTSmallStudent,
    build_student_from_config,
    count_parameters,
)


def test_student_model_shape_norm_and_param_budget() -> None:
    model = MelCnnStudent()
    n_params = count_parameters(model)
    assert STUDENT_PARAM_MIN <= n_params <= STUDENT_PARAM_MAX

    x = torch.randn(2, 1, 96, 251)
    y = model(x)
    assert y.shape == (2, 512)
    norms = y.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_passt_small_model_shape_norm_and_param_budget() -> None:
    model = PaSSTSmallStudent()
    n_params = count_parameters(model)
    assert STUDENT_PARAM_MIN <= n_params <= STUDENT_PARAM_MAX

    x = torch.randn(2, 1, 96, 251)
    y = model(x)
    assert y.shape == (2, 512)
    norms = y.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_build_student_from_passt_config() -> None:
    model = PaSSTSmallStudent(patchout_freq=0, patchout_time=0)
    rebuilt = build_student_from_config(model.config_dict())
    assert isinstance(rebuilt, PaSSTSmallStudent)
    assert rebuilt.patchout_freq == 0
    assert rebuilt.patchout_time == 0


def test_log_mel_extractor_shape() -> None:
    mel = LogMelExtractor(MelConfig())
    wav = torch.zeros(2, WINDOW_SAMPLES)
    out = mel(wav)
    assert out.shape == (2, 1, 96, 251)
    assert torch.isfinite(out).all()


def test_ensure_window_length_pads_and_trims() -> None:
    short = torch.ones(10)
    padded = ensure_window_length(short, 12)
    assert padded.shape == (1, 12)
    assert padded[0, :10].sum() == 10
    assert padded[0, 10:].sum() == 0

    long = ensure_window_length(torch.arange(20), 8)
    assert long.shape == (1, 8)
    torch.testing.assert_close(long[0], torch.arange(8))


def test_window_starts_cover_track() -> None:
    starts = window_starts(300_000, windows_per_track=3, window_samples=120_000)
    assert starts == [0, 90_000, 180_000]
    assert window_starts(10, windows_per_track=2, window_samples=120_000) == [0, 0]


def test_topk_overlap_identity_is_one() -> None:
    x = np.eye(4, dtype=np.float32)
    out = topk_overlap(x, x, k_values=(1, 2))
    assert out == {1: 1.0, 2: 1.0}


def test_resolve_torch_device_cpu_and_auto() -> None:
    assert resolve_torch_device("cpu") == torch.device("cpu")
    assert resolve_torch_device("auto").type in {"cpu", "cuda", "mps"}


def test_distill_target_dataset_can_cache_waveforms(monkeypatch) -> None:
    frame = pd.DataFrame(
        [
            {
                "path": "x.mp3",
                "start_sample": 0,
                "duration_samples": 4,
                "teacher_embedding": np.ones(2, dtype=np.float32),
                "genre_top": "Rock",
            }
        ]
    )

    calls = 0

    def fake_load_window(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return np.arange(4, dtype=np.float32)

    monkeypatch.setattr("student.data.load_window", fake_load_window)
    dataset = DistillTargetDataset(frame, cache_waveforms=True)
    wav, target, label = dataset[0]
    wav_again, _, _ = dataset[0]

    assert calls == 1
    torch.testing.assert_close(wav, torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(wav_again, wav)
    torch.testing.assert_close(target, torch.ones(2))
    assert label == "Rock"
