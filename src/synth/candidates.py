"""Candidate table: 960-d embeddings + track_row aligned to the Plan 1 manifest."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

_EMBED_DIM = 960
_META_COLS = ["song_id", "title", "artist", "album", "genre", "year", "bpm", "energy"]


def load_embeddings(playback_db: str) -> dict[str, np.ndarray]:
    """Decode ``TrackEmbeddingEntity.embedding`` (little-endian float32, 960 dims) by ``songUid``."""
    con = sqlite3.connect(f"file:{playback_db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT songUid, embedding FROM TrackEmbeddingEntity").fetchall()
    finally:
        con.close()
    out: dict[str, np.ndarray] = {}
    for song_id, blob in rows:
        vec = np.frombuffer(blob, dtype="<f4", count=_EMBED_DIM).astype(np.float32)
        out[song_id] = vec
    return out


@dataclass
class CandidateTable:
    song_ids: list[str]
    track_row: dict[str, int]
    matrix: np.ndarray
    meta: pd.DataFrame


def build_candidate_table(manifest: pd.DataFrame, playback_db: str) -> CandidateTable:
    """Join ``manifest`` to embeddings by ``song_id``.

    Drops tracks with no embedding, deduplicates ``song_id`` (keep first), and
    assigns ``track_row = 0..N-1`` matching the resulting ``matrix``/``meta``.
    """
    emb = load_embeddings(playback_db)
    seen: set[str] = set()
    kept_rows, vectors = [], []
    for r in manifest.itertuples(index=False):
        sid = r.song_id
        if sid in seen or sid not in emb:
            continue
        seen.add(sid)
        kept_rows.append(r)
        vectors.append(emb[sid])
    matrix = np.vstack(vectors).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    meta = pd.DataFrame([{c: getattr(r, c) for c in _META_COLS} for r in kept_rows])
    song_ids = list(meta["song_id"])
    return CandidateTable(
        song_ids=song_ids,
        track_row={sid: i for i, sid in enumerate(song_ids)},
        matrix=matrix,
        meta=meta.reset_index(drop=True),
    )
