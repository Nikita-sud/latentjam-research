import sqlite3
import struct

import numpy as np
import pandas as pd

from synth.candidates import build_candidate_table, load_embeddings


def _emb_blob(vec):
    return struct.pack(f"<{len(vec)}f", *vec)


def _make_playback(path, rows):  # rows: list[(song_id, vec)]
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE TrackEmbeddingEntity (songUid TEXT PRIMARY KEY, modelVersion TEXT, "
                "embedding BLOB, embeddedAtMs INTEGER, tempo REAL, energy REAL)")
    for sid, vec in rows:
        con.execute("INSERT INTO TrackEmbeddingEntity VALUES (?,?,?,?,?,?)",
                    (sid, "v3", _emb_blob(vec), 0, 90.0, 0.5))
    con.commit()
    con.close()


def test_load_embeddings_roundtrip(tmp_path):
    db = tmp_path / "p.db"
    vec = list(np.arange(960, dtype=np.float32) / 960.0)
    _make_playback(db, [("uas-a", vec)])
    emb = load_embeddings(str(db))
    assert emb["uas-a"].shape == (960,)
    assert np.allclose(emb["uas-a"], vec, atol=1e-6)


def test_build_candidate_table_aligns_and_dedupes(tmp_path):
    db = tmp_path / "p.db"

    def v(k):
        return list((np.arange(960, dtype=np.float32) + k) / 960.0)

    _make_playback(db, [("uas-a", v(0)), ("uas-b", v(1))])
    manifest = pd.DataFrame({
        "song_id": ["uas-a", "uas-b", "uas-b", "uas-z"],  # dup uas-b; uas-z has no embedding
        "title": ["A", "B", "B", "Z"], "artist": ["x", "y", "y", "z"],
        "album": [None]*4, "genre": ["Pop", "Rock", "Rock", "Jazz"],
        "year": [2000]*4, "language": [None]*4, "bpm": [90.0]*4, "energy": [0.5]*4,
    })
    ct = build_candidate_table(manifest, str(db))
    assert ct.song_ids == ["uas-a", "uas-b"]                 # dedup + drop no-embedding
    assert ct.matrix.shape == (2, 960)
    assert np.allclose(np.linalg.norm(ct.matrix, axis=1), 1.0, atol=1e-5)  # L2-normed
    assert ct.track_row == {"uas-a": 0, "uas-b": 1}
    assert list(ct.meta["song_id"]) == ["uas-a", "uas-b"]
