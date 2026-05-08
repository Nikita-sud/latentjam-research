"""Content-only retrieval metrics for v0.

Without listening history we can't measure recommendation quality directly, so
we use proxies that say "embeddings put similar music near each other":

- ``recall_at_k_genre``: fraction of top-k neighbors sharing the seed's genre.
- ``tag_jaccard_at_k``: mean Jaccard between the seed's tag set and the top-k
  neighbors' tag sets (richer than genre on MagnaTagATune).
- ``album_cohesion``: intra-album mean cosine vs. cross-album mean cosine on
  the user's own collection. A useful sanity check on real-world data.

All inputs are assumed L2-normalized along the embedding axis (so dot product
== cosine).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _dot_topk(matrix: np.ndarray, query: np.ndarray, k: int, exclude_self: int) -> np.ndarray:
    scores = matrix @ query
    if exclude_self >= 0:
        scores[exclude_self] = -np.inf
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    part = np.argpartition(-scores, k)[:k]
    return part[np.argsort(-scores[part])]


def recall_at_k_genre(
    matrix: np.ndarray,
    labels: Sequence[str | int],
    k_values: Sequence[int],
) -> dict[int, float]:
    """Mean recall@k over all rows, where a hit = neighbor shares the seed's label.

    Returns ``{k: recall_value}`` for each ``k`` in ``k_values``. Random
    baseline for ``L`` balanced classes is ``1 / L``.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {matrix.shape}")
    n = matrix.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != matrix rows {n}")

    labels_arr = np.asarray(labels)
    k_max = max(k_values)
    out = {k: 0.0 for k in k_values}

    for i in range(n):
        idx = _dot_topk(matrix, matrix[i], k=k_max, exclude_self=i)
        same = (labels_arr[idx] == labels_arr[i]).astype(np.float32)
        for k in k_values:
            out[k] += float(same[:k].mean())

    return {k: v / n for k, v in out.items()}


def tag_jaccard_at_k(
    matrix: np.ndarray,
    tag_matrix: np.ndarray,
    k_values: Sequence[int],
) -> dict[int, float]:
    """Mean tag-Jaccard@k over all rows.

    ``tag_matrix`` is shape ``(N, T)`` boolean (or 0/1). For each row, we take
    its top-k cosine neighbors and compute the mean Jaccard between the seed's
    tag set and each neighbor's tag set. Rows with no tags are skipped.
    """
    if matrix.ndim != 2 or tag_matrix.ndim != 2:
        raise ValueError("matrix and tag_matrix must be 2-D")
    n = matrix.shape[0]
    if tag_matrix.shape[0] != n:
        raise ValueError(
            f"tag_matrix rows {tag_matrix.shape[0]} != matrix rows {n}"
        )

    tags = tag_matrix.astype(bool)
    counted = 0
    sums = {k: 0.0 for k in k_values}
    k_max = max(k_values)

    for i in range(n):
        seed_tags = tags[i]
        if not seed_tags.any():
            continue
        idx = _dot_topk(matrix, matrix[i], k=k_max, exclude_self=i)
        # Boolean per-pair Jaccard.
        for k in k_values:
            top = idx[:k]
            inter = (tags[top] & seed_tags).sum(axis=1)
            union = (tags[top] | seed_tags).sum(axis=1).clip(min=1)
            sums[k] += float((inter / union).mean())
        counted += 1

    if counted == 0:
        return {k: 0.0 for k in k_values}
    return {k: v / counted for k, v in sums.items()}


def album_cohesion(
    matrix: np.ndarray,
    album_ids: Sequence[str | int],
    min_tracks: int = 4,
) -> dict[str, float]:
    """Compare intra-album vs cross-album mean cosine.

    Returns ``{"intra_mean": ..., "cross_mean": ..., "gap": intra - cross,
    "n_albums": ..., "n_tracks": ...}``. Gap > 0 means embeddings cluster by
    album, which is a useful sanity check on the user's own library.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {matrix.shape}")
    n = matrix.shape[0]
    if len(album_ids) != n:
        raise ValueError(f"album_ids length {len(album_ids)} != matrix rows {n}")

    by_album: dict[str | int, list[int]] = {}
    for i, aid in enumerate(album_ids):
        if aid is None or (isinstance(aid, float) and np.isnan(aid)):
            continue
        by_album.setdefault(aid, []).append(i)

    intra_scores: list[float] = []
    n_albums = 0
    n_tracks = 0
    selected_rows: list[int] = []

    for _aid, rows in by_album.items():
        if len(rows) < min_tracks:
            continue
        n_albums += 1
        n_tracks += len(rows)
        sub = matrix[rows]
        sims = sub @ sub.T
        # Off-diagonal entries only.
        m = sims.shape[0]
        mask = ~np.eye(m, dtype=bool)
        intra_scores.append(float(sims[mask].mean()))
        selected_rows.extend(rows)

    if n_albums == 0:
        return {
            "intra_mean": 0.0,
            "cross_mean": 0.0,
            "gap": 0.0,
            "n_albums": 0,
            "n_tracks": 0,
        }

    rng = np.random.default_rng(0)
    selected = np.array(selected_rows, dtype=np.int64)
    cross_pairs = min(20_000, len(selected) * (len(selected) - 1) // 2)
    cross_scores: list[float] = []
    for _ in range(cross_pairs):
        a, b = rng.choice(len(selected), size=2, replace=False)
        if album_ids[selected[a]] == album_ids[selected[b]]:
            continue
        cross_scores.append(float(matrix[selected[a]] @ matrix[selected[b]]))

    intra_mean = float(np.mean(intra_scores))
    cross_mean = float(np.mean(cross_scores)) if cross_scores else 0.0
    return {
        "intra_mean": intra_mean,
        "cross_mean": cross_mean,
        "gap": intra_mean - cross_mean,
        "n_albums": n_albums,
        "n_tracks": n_tracks,
    }
