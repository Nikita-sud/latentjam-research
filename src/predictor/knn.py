"""Pure-numpy cosine top-k search over a row-aligned embedding matrix.

Both the query and the matrix rows are assumed to be L2-normalized; under that
assumption the dot product equals cosine similarity. ``MertEncoder`` already
produces L2-normalized vectors, so callers don't need to renormalize.

Why no faiss in v0: a (100_000, 768) float32 matrix is ~300 MB, and a single
matvec is ~30 ms on a laptop CPU. faiss buys nothing at this scale and adds a
heavy native dep. It's available behind the ``recommend`` extras for later.
"""

from __future__ import annotations

import numpy as np


def cosine_topk(
    query: np.ndarray,
    matrix: np.ndarray,
    k: int,
    exclude: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(indices, scores)`` of the top-``k`` cosine matches.

    ``query`` is shape ``(D,)``; ``matrix`` is shape ``(N, D)``. Both must be
    L2-normalized along ``D``. Indices in ``exclude`` are masked to ``-inf``
    before the top-k pass.
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if query.ndim != 1 or query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"query shape {query.shape} incompatible with matrix shape {matrix.shape}"
        )

    n = matrix.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    k_eff = min(k, n)
    if k_eff == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    scores = matrix @ query.astype(np.float32, copy=False)

    if exclude:
        mask = np.fromiter(
            (i in exclude for i in range(n)), count=n, dtype=bool
        )
        scores = np.where(mask, -np.inf, scores)

    if k_eff >= n:
        order = np.argsort(-scores)
    else:
        # argpartition gives unsorted top-k_eff; then sort just those.
        part = np.argpartition(-scores, k_eff - 1)[:k_eff]
        order = part[np.argsort(-scores[part])]

    return order.astype(np.int64), scores[order].astype(np.float32)
