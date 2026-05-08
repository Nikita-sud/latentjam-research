"""Metrics for student-vs-teacher embedding comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def relational_distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    """MSE between in-batch cosine similarity matrices."""
    if student_embeddings.shape[0] < 2:
        return student_embeddings.new_tensor(0.0)
    s_sim = student_embeddings @ student_embeddings.T
    t_sim = teacher_embeddings @ teacher_embeddings.T
    mask = ~torch.eye(s_sim.shape[0], dtype=torch.bool, device=s_sim.device)
    return nn.functional.mse_loss(s_sim[mask], t_sim[mask])


def cosine_summary(student: np.ndarray, teacher: np.ndarray) -> dict[str, float]:
    cos = np.sum(student * teacher, axis=1)
    return {
        "cosine_mean": float(cos.mean()),
        "cosine_min": float(cos.min()),
        "cosine_p25": float(np.percentile(cos, 25)),
        "cosine_p50": float(np.percentile(cos, 50)),
        "cosine_p75": float(np.percentile(cos, 75)),
    }


def topk_overlap(
    student: np.ndarray,
    teacher: np.ndarray,
    *,
    k_values: Sequence[int] = (5, 10, 20),
) -> dict[int, float]:
    """Mean overlap between student and teacher nearest-neighbor sets."""
    if student.shape != teacher.shape:
        raise ValueError(f"shape mismatch: {student.shape} != {teacher.shape}")
    n = student.shape[0]
    if n < 2:
        return {int(k): 0.0 for k in k_values}
    max_k = min(max(k_values), n - 1)
    s_scores = student @ student.T
    t_scores = teacher @ teacher.T
    np.fill_diagonal(s_scores, -np.inf)
    np.fill_diagonal(t_scores, -np.inf)
    s_rank = np.argsort(-s_scores, axis=1)[:, :max_k]
    t_rank = np.argsort(-t_scores, axis=1)[:, :max_k]
    out: dict[int, float] = {}
    for k in k_values:
        kk = min(int(k), max_k)
        if kk <= 0:
            out[int(k)] = 0.0
            continue
        total = 0.0
        for i in range(n):
            total += len(set(s_rank[i, :kk]).intersection(t_rank[i, :kk])) / float(kk)
        out[int(k)] = total / n
    return out
