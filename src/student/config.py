"""Shared constants for the CLAP-distilled mel-CNN student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Final

TARGET_SR: Final[int] = 24_000
CLAP_SR: Final[int] = 48_000
WINDOW_SECONDS: Final[float] = 5.0
WINDOW_SAMPLES: Final[int] = int(TARGET_SR * WINDOW_SECONDS)
CLAP_WINDOW_SAMPLES: Final[int] = int(CLAP_SR * WINDOW_SECONDS)

TEACHER_EMBEDDING_DIM: Final[int] = 512
STUDENT_PARAM_MIN: Final[int] = 5_000_000
STUDENT_PARAM_MAX: Final[int] = 15_000_000
STUDENT_MODEL_VERSION: Final[str] = "mel-cnn-clap@96mel-5s/512d/v1"

DEFAULT_CLAP_MUSIC_CKPT: Final[str] = (
    "models/clap/music_audioset_epoch_15_esc_90.14.pt"
)
DEFAULT_FMA_TARGETS: Final[str] = "models/student/fma_small_clap_targets.parquet"
DEFAULT_STUDENT_CKPT: Final[str] = "models/student/mel_cnn_student.pt"


@dataclass(frozen=True)
class MelConfig:
    sample_rate: int = TARGET_SR
    window_samples: int = WINDOW_SAMPLES
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 480
    n_mels: int = 96
    f_min: float = 20.0
    f_max: float | None = 12_000.0
    amin: float = 1e-5
    normalize: bool = True

    def to_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)
