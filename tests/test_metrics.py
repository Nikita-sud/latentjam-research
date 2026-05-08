from __future__ import annotations

import numpy as np

from eval.metrics import album_cohesion, recall_at_k_genre, tag_jaccard_at_k


def _normed(rows: list[list[float]]) -> np.ndarray:
    m = np.array(rows, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-12)
    return m


def test_recall_at_k_perfect_when_clusters_separate() -> None:
    matrix = _normed(
        [
            [10.0, 0.0],
            [9.0, 0.5],
            [-10.0, 0.0],
            [-9.0, 0.5],
        ]
    )
    labels = ["a", "a", "b", "b"]
    result = recall_at_k_genre(matrix, labels, k_values=(1,))
    assert result[1] == 1.0


def test_recall_at_k_random_when_labels_random() -> None:
    rng = np.random.default_rng(0)
    matrix = _normed(rng.standard_normal((50, 16)).tolist())
    labels = rng.integers(0, 4, size=50).tolist()
    result = recall_at_k_genre(matrix, labels, k_values=(5,))
    # Random expectation is 0.25; allow generous slack on small n.
    assert 0.05 < result[5] < 0.5


def test_recall_at_k_validates_inputs() -> None:
    matrix = _normed([[1.0, 0.0], [0.0, 1.0]])
    try:
        recall_at_k_genre(matrix, ["a"], k_values=(1,))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_tag_jaccard_perfect_when_tags_match() -> None:
    matrix = _normed(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ]
    )
    tags = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    result = tag_jaccard_at_k(matrix, tags, k_values=(1,))
    assert result[1] == 1.0


def test_tag_jaccard_skips_empty_tag_rows() -> None:
    matrix = _normed([[1.0, 0.0], [0.0, 1.0]])
    tags = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    result = tag_jaccard_at_k(matrix, tags, k_values=(1,))
    # Only the second row contributes; its top-1 neighbor has empty tags, so
    # Jaccard = 0 / |seed_tags| = 0.
    assert result[1] == 0.0


def test_album_cohesion_positive_gap_when_clustered() -> None:
    rng = np.random.default_rng(0)
    base_a = rng.standard_normal(8)
    base_b = -rng.standard_normal(8)

    rows = []
    aids = []
    for _ in range(6):
        rows.append(base_a + 0.05 * rng.standard_normal(8))
        aids.append("A")
    for _ in range(6):
        rows.append(base_b + 0.05 * rng.standard_normal(8))
        aids.append("B")
    matrix = _normed(rows)
    out = album_cohesion(matrix, aids, min_tracks=4)
    assert out["n_albums"] == 2
    assert out["gap"] > 0.5


def test_album_cohesion_handles_no_eligible_albums() -> None:
    matrix = _normed([[1.0, 0.0], [0.0, 1.0]])
    out = album_cohesion(matrix, ["A", "A"], min_tracks=4)
    assert out["n_albums"] == 0
    assert out["gap"] == 0.0
