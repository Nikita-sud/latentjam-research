from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from predictor.store import EmbeddingStore, TrackRecord


def _norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)


def _record(track_id: str, vec: np.ndarray, **kw) -> TrackRecord:
    return TrackRecord(
        track_id=track_id,
        path=f"/tmp/{track_id}.flac",
        embedding=_norm(vec.astype(np.float32)),
        model_version="mert-v1-95m@layers5-8/meanpool/v1",
        **kw,
    )


def test_add_and_search(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "x.parquet")
    rng = np.random.default_rng(0)

    for i in range(5):
        store.add(_record(f"id{i}", rng.standard_normal(16), title=f"Track {i}"))

    hits = store.search_by_id("id2", k=3)
    assert hits[0].track_id != "id2"  # excluded
    assert len(hits) == 3


def test_dim_mismatch_rejected(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "x.parquet")
    store.add(_record("a", np.ones(16)))
    with pytest.raises(ValueError):
        store.add(_record("b", np.ones(8)))


def test_save_and_reopen_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "store.parquet"
    rng = np.random.default_rng(7)

    store = EmbeddingStore.open(p)
    for i in range(4):
        store.add(
            _record(
                f"id{i}",
                rng.standard_normal(8),
                title=f"T{i}",
                artist="An Artist",
                album="An Album",
                genre="rock",
                year=2024,
            )
        )
    store.save(p)
    assert p.exists()
    assert p.with_suffix(".npy").exists()

    reopened = EmbeddingStore.open(p)
    assert len(reopened) == 4
    assert reopened.dim == 8
    for i in range(4):
        emb_a = store.get_embedding(f"id{i}")
        emb_b = reopened.get_embedding(f"id{i}")
        np.testing.assert_allclose(emb_a, emb_b, atol=1e-6)


def test_update_existing_track(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "x.parquet")
    store.add(_record("a", np.array([1.0, 0.0])))
    store.add(_record("a", np.array([0.0, 1.0])))  # overwrite
    assert len(store) == 1
    np.testing.assert_allclose(store.get_embedding("a"), np.array([0.0, 1.0]))


def test_search_returns_empty_for_empty_store(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "x.parquet")
    hits = store.search(np.ones(8, dtype=np.float32) / np.sqrt(8), k=5)
    assert hits == []


def test_contains(tmp_path: Path) -> None:
    store = EmbeddingStore.open(tmp_path / "x.parquet")
    store.add(_record("a", np.ones(4)))
    assert store.contains("a")
    assert not store.contains("nope")
