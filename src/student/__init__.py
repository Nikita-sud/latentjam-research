"""Distilled on-device student models."""

from student.config import STUDENT_MODEL_VERSION
from student.model import MelCnnStudent, count_parameters

__all__ = ["MelCnnStudent", "STUDENT_MODEL_VERSION", "count_parameters"]
