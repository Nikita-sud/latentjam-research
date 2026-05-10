"""Distilled on-device student models."""

from student.config import STUDENT_MODEL_VERSION
from student.model import (
    MelCnnStudent,
    PaSSTSmallStudent,
    build_student_from_config,
    build_student_model,
    count_parameters,
)

__all__ = [
    "MelCnnStudent",
    "PaSSTSmallStudent",
    "STUDENT_MODEL_VERSION",
    "build_student_from_config",
    "build_student_model",
    "count_parameters",
]
