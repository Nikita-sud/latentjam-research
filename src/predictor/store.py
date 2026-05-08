"""Parquet-backed embedding store with an in-memory matrix view.

Schema (see ``CLAUDE.md`` for the canonical column list):

| Column          | Type                  |
|-----------------|-----------------------|
| track_id        | string (xxhash64 hex) |
| path            | string                |
| title           | string \\| null       |
| artist          | string \\| null       |
| album           | string \\| null       |
| genre           | string \\| null       |
| year            | int32 \\| null        |
| model_version   | string                |
| embedding_dim   | int32                 |
| embedding       | list<float32>         |

A sidecar ``embeddings.npy`` is written next to the parquet on ``save()``; on
``open()`` it is reused if newer than the parquet, otherwise rebuilt from the
parquet's ``embedding`` column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from predictor.knn import cosine_topk

_META_COLS: tuple[str, ...] = (
    "track_id",
    "path",
    "title",
    "artist",
    "album",
    "genre",
    "year",
    "model_version",
    "embedding_dim",
)


@dataclass
class TrackRecord:
    track_id: str
    path: str
    embedding: np.ndarray  # float32, L2-normalized
    model_version: str
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    year: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    rank: int
    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    genre: str | None
    score: float


class EmbeddingStore:
    """In-memory metadata DataFrame + (N, D) embedding matrix, parquet-persisted.

    Designed for batch ingestion: ``add()`` is fast (no recompute of the matrix
    until ``save()`` or ``rebuild_matrix()``). ``search()`` requires the matrix
    to be current — it auto-rebuilds if rows have changed.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame = self._empty_frame()
        self._matrix: np.ndarray | None = None
        self._dim: int | None = None
        self._dirty: bool = False
        self._id_to_row: dict[str, int] = {}

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "track_id": pd.Series(dtype="string"),
                "path": pd.Series(dtype="string"),
                "title": pd.Series(dtype="string"),
                "artist": pd.Series(dtype="string"),
                "album": pd.Series(dtype="string"),
                "genre": pd.Series(dtype="string"),
                "year": pd.Series(dtype="Int32"),
                "model_version": pd.Series(dtype="string"),
                "embedding_dim": pd.Series(dtype="Int32"),
                "embedding": pd.Series(dtype="object"),
            }
        )
        return df

    @classmethod
    def open(cls, path: str | Path) -> EmbeddingStore:
        """Load an existing store, or return an empty one if the parquet is missing."""
        store = cls()
        p = Path(path)
        if p.exists():
            store._df = pd.read_parquet(p)
            store._df = store._df.reindex(
                columns=list(_META_COLS) + ["embedding"]
            )
            if not store._df.empty:
                # Embeddings come back as object column of np.ndarrays (or lists).
                store._df["embedding"] = store._df["embedding"].apply(
                    lambda v: np.asarray(v, dtype=np.float32)
                )
                store._dim = int(store._df["embedding_dim"].iloc[0])
            store._rebuild_index()

            sidecar = p.with_suffix(".npy")
            if sidecar.exists() and sidecar.stat().st_mtime >= p.stat().st_mtime:
                m = np.load(sidecar)
                if m.shape == (len(store._df), store._dim or 0):
                    store._matrix = m.astype(np.float32, copy=False)
                else:
                    store.rebuild_matrix()
            else:
                store.rebuild_matrix()
        return store

    def __len__(self) -> int:
        return len(self._df)

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def df(self) -> pd.DataFrame:
        """Read-only view of the metadata frame (callers should not mutate)."""
        return self._df

    def contains(self, track_id: str) -> bool:
        return track_id in self._id_to_row

    def add(self, record: TrackRecord) -> None:
        if record.embedding.dtype != np.float32:
            record.embedding = record.embedding.astype(np.float32, copy=False)
        if record.embedding.ndim != 1:
            raise ValueError(
                f"embedding must be 1-D, got shape {record.embedding.shape}"
            )

        d = int(record.embedding.shape[0])
        if self._dim is None:
            self._dim = d
        elif d != self._dim:
            raise ValueError(
                f"embedding_dim mismatch: store has {self._dim}, got {d}"
            )

        row = {
            "track_id": record.track_id,
            "path": record.path,
            "title": record.title,
            "artist": record.artist,
            "album": record.album,
            "genre": record.genre,
            "year": record.year,
            "model_version": record.model_version,
            "embedding_dim": d,
            "embedding": record.embedding,
        }

        if record.track_id in self._id_to_row:
            idx = self._id_to_row[record.track_id]
            for k, v in row.items():
                self._df.at[idx, k] = v
        else:
            self._df.loc[len(self._df)] = row
            self._id_to_row[record.track_id] = len(self._df) - 1

        self._dirty = True

    def get_embedding(self, track_id: str) -> np.ndarray:
        if track_id not in self._id_to_row:
            raise KeyError(f"unknown track_id: {track_id}")
        return np.asarray(self._df.at[self._id_to_row[track_id], "embedding"], dtype=np.float32)

    def rebuild_matrix(self) -> None:
        if self._df.empty or self._dim is None:
            self._matrix = None
            self._dirty = False
            return
        embs = self._df["embedding"].to_list()
        self._matrix = np.stack(
            [np.asarray(e, dtype=np.float32) for e in embs], axis=0
        )
        self._dirty = False

    def _rebuild_index(self) -> None:
        self._id_to_row = {
            tid: i for i, tid in enumerate(self._df["track_id"].to_list())
        }

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        if self._dirty or self._matrix is None:
            self.rebuild_matrix()
        if self._matrix is None:
            return []

        exclude_rows: set[int] = set()
        if exclude_ids:
            for tid in exclude_ids:
                if tid in self._id_to_row:
                    exclude_rows.add(self._id_to_row[tid])

        idxs, scores = cosine_topk(query, self._matrix, k=k, exclude=exclude_rows or None)

        hits: list[SearchHit] = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), scores.tolist(), strict=True)):
            row = self._df.iloc[i]
            hits.append(
                SearchHit(
                    rank=rank,
                    track_id=row["track_id"],
                    path=row["path"],
                    title=row.get("title"),
                    artist=row.get("artist"),
                    album=row.get("album"),
                    genre=row.get("genre"),
                    score=float(s),
                )
            )
        return hits

    def search_by_id(
        self, track_id: str, k: int = 10, exclude_self: bool = True
    ) -> list[SearchHit]:
        query = self.get_embedding(track_id)
        excl = {track_id} if exclude_self else None
        return self.search(query, k=k, exclude_ids=excl)

    def save(self, path: str | Path, write_sidecar: bool = True) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Embeddings round-trip as parquet ``list<float>``.
        df_out = self._df.copy()
        df_out["embedding"] = df_out["embedding"].apply(
            lambda v: np.asarray(v, dtype=np.float32).tolist() if v is not None else None
        )
        df_out.to_parquet(p, index=False)

        if write_sidecar:
            self.rebuild_matrix()
            if self._matrix is not None:
                np.save(p.with_suffix(".npy"), self._matrix.astype(np.float32, copy=False))
        self._dirty = False
