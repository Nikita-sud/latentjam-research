from __future__ import annotations

import numpy as np
import pytest

from predictor.knn import cosine_topk


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)


def test_cosine_topk_ranks_identical_first() -> None:
    rng = np.random.default_rng(0)
    matrix = _normalize(rng.standard_normal((10, 8)).astype(np.float32))
    query = matrix[3].copy()

    idx, scores = cosine_topk(query, matrix, k=5)
    assert idx[0] == 3
    assert pytest.approx(scores[0], abs=1e-5) == 1.0
    # Strictly decreasing scores.
    diffs = np.diff(scores)
    assert (diffs <= 1e-6).all()


def test_cosine_topk_excludes_indices() -> None:
    rng = np.random.default_rng(1)
    matrix = _normalize(rng.standard_normal((10, 8)).astype(np.float32))
    query = matrix[3].copy()

    idx, _ = cosine_topk(query, matrix, k=3, exclude={3})
    assert 3 not in idx.tolist()
    assert len(idx) == 3


def test_cosine_topk_k_larger_than_n() -> None:
    matrix = _normalize(np.eye(3, dtype=np.float32))
    query = matrix[0]
    idx, scores = cosine_topk(query, matrix, k=10)
    assert idx.shape == (3,)
    assert scores.shape == (3,)


def test_cosine_topk_rejects_shape_mismatch() -> None:
    matrix = np.zeros((4, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_topk(np.zeros(7, dtype=np.float32), matrix, k=2)
    with pytest.raises(ValueError):
        cosine_topk(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), k=2)
    with pytest.raises(ValueError):
        cosine_topk(np.zeros(8, dtype=np.float32), matrix, k=0)
