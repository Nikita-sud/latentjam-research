"""Cross-module seam test: engagement.load_real_events -> validate.validate_corpus.

``load_real_events`` reads ``ListeningEventEntity`` as-is (id column ``songUid``),
while ``validate_corpus`` looks up ``real_events["song_id"]``. Before the fix,
wiring these two functions together (exactly as the corpus-validation pipeline
does) raised ``KeyError: 'song_id'``. This test builds a real sqlite
``ListeningEventEntity`` table and drives both functions end-to-end to guard the
seam directly, rather than only unit-testing each function in isolation.
"""

import sqlite3

import numpy as np
import pandas as pd

from synth.engagement import load_real_events
from synth.validate import ValidationReport, validate_corpus


def _manifest():
    genres = ["Anime OST"] * 5 + ["Hip-Hop"] * 3 + ["Disco"] * 2
    return pd.DataFrame({"song_id": [f"s{i}" for i in range(10)], "genre": genres})


def _make_listening_event_db(path, manifest_ids, n=400, completion=0.44, seed=0):
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE ListeningEventEntity ("
        "songUid TEXT, completed INTEGER, playedMs INTEGER, "
        "trackDurationMs INTEGER, finalizeReason TEXT)"
    )
    completed = rng.random(n) < completion
    song_uids = rng.choice(manifest_ids, n)
    for uid, comp in zip(song_uids, completed, strict=True):
        played = 200_000 if comp else int(rng.integers(1000, 60000))
        reason = "TRACK_ENDED" if comp else "USER_SKIPPED"
        con.execute(
            "INSERT INTO ListeningEventEntity VALUES (?, ?, ?, ?, ?)",
            (uid, int(comp), played, 210_000, reason),
        )
    con.commit()
    con.close()


def test_load_real_events_renames_songuid_to_song_id(tmp_path):
    manifest = _manifest()
    db_path = tmp_path / "listening_events.db"
    _make_listening_event_db(db_path, manifest["song_id"].tolist())

    real = load_real_events(str(db_path))

    assert "song_id" in real.columns
    assert "songUid" not in real.columns


def test_load_real_events_then_validate_corpus_does_not_raise_keyerror(tmp_path):
    # This is the exact seam that was broken: wiring load_real_events straight
    # into validate_corpus's `real_events["song_id"]` lookup.
    manifest = _manifest()
    db_path = tmp_path / "listening_events.db"
    _make_listening_event_db(db_path, manifest["song_id"].tolist())

    real = load_real_events(str(db_path))

    rng = np.random.default_rng(1)
    ids = rng.choice(manifest["song_id"], 800)  # broad coverage, similar genre mix
    sessions = pd.DataFrame({"song_id": ids, "completed": (rng.random(800) < 0.44).astype(int)})

    report = validate_corpus(sessions, manifest, real)

    assert isinstance(report, ValidationReport)


def test_load_real_events_leaves_already_song_id_csv_untouched(tmp_path):
    # A CSV that already uses `song_id` (no `songUid` column) must pass through
    # unchanged -- no accidental renaming, no dropped/added columns.
    csv_path = tmp_path / "events.csv"
    pd.DataFrame({"song_id": ["s0", "s1"], "completed": [1, 0]}).to_csv(csv_path, index=False)

    real = load_real_events(str(csv_path))

    assert list(real.columns) == ["song_id", "completed"]
    assert real["song_id"].tolist() == ["s0", "s1"]
