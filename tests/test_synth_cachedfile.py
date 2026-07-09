import json
import os
import sqlite3
from pathlib import Path

import pytest

from synth.cachedfile import (
    directory_name,
    song_id_from_cached_file,
    split_multi,
    valid_mbid,
)
from synth.uid import song_uid_mbid

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cachedfile_uid_vectors.json"

_URI = (
    "content://com.android.externalstorage.documents/tree/primary%3AMusic/"
    "document/primary%3AMusic%2F{}"
)


def _load_vectors():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def test_fixture_reproduces_real_ondevice_uids():
    """Durable guard: every committed CachedFileData vector reproduces its real songUid.

    Each ``expected_song_uid`` is a real id from ``TrackEmbeddingEntity.songUid`` on the
    device (see tests/fixtures/cachedfile_uid_vectors.json, generated from music_cache.db
    cross-checked against the playback DB). The vectors deliberately span the MBID path,
    the null-album directory fallback, escaped ``\\;`` multi-values, and null date/track/disc.
    """
    vectors = _load_vectors()
    assert len(vectors) >= 40, f"expected >=40 fixture vectors, got {len(vectors)}"
    mismatches = [
        (v["expected_song_uid"], got)
        for v in vectors
        if (got := song_id_from_cached_file(v)) != v["expected_song_uid"]
    ]
    assert not mismatches, f"{len(mismatches)}/{len(vectors)} vectors mismatched: {mismatches[:5]}"


def test_fixture_covers_the_tricky_cases():
    """The fixture must actually exercise each hard path, or it is not a real guard."""
    vectors = _load_vectors()
    assert any(v["expected_song_uid"].startswith("ums") for v in vectors), "no MBID-path vector"
    assert any(v["expected_song_uid"].startswith("uas") for v in vectors), "no auxio-path vector"
    assert any(v["albumName"] is None for v in vectors), "no null-album (dir-fallback) vector"
    assert any(
        "\\;" in (v.get("artistNames") or "") or "\\;" in (v.get("albumArtistNames") or "")
        for v in vectors
    ), "no escaped-semicolon vector"
    assert any(
        v["date"] is None and v["track"] is None and v["disc"] is None for v in vectors
    ), "no null date/track/disc vector"


# --- helper unit tests ------------------------------------------------------


def test_split_multi_handles_escaping():
    assert split_multi(None) == []
    assert split_multi("") == []
    assert split_multi("solo") == ["solo"]
    assert split_multi("Dio;Jotaro [AI Cover]") == ["Dio", "Jotaro [AI Cover]"]
    # An escaped \; is a literal ';' inside a single value, not a delimiter.
    assert split_multi("Mihail Krug\\; Mihail G") == ["Mihail Krug; Mihail G"]
    assert split_multi("a\\;b;c") == ["a;b", "c"]


def test_directory_name_from_saf_uri():
    assert directory_name(_URI.format("file.mp3")) == "Music"
    assert directory_name(_URI.format("%D0%A2%D0%B5%D1%81%D1%82.opus")) == "Music"
    # Nested subdirectory: parent component wins.
    nested = (
        "content://com.android.externalstorage.documents/tree/primary%3AMusic/"
        "document/primary%3AMusic%2FSubDir%2Fsong.flac"
    )
    assert directory_name(nested) == "SubDir"


def test_valid_mbid():
    assert valid_mbid("69b5fc2d-1de1-3b2d-8816-0ad90ee89903") == "69b5fc2d-1de1-3b2d-8816-0ad90ee89903"
    assert valid_mbid(None) is None
    assert valid_mbid("") is None
    assert valid_mbid("not-a-uuid") is None


def test_song_id_from_cached_file_selects_mbid_vs_auxio():
    mbid = "69b5fc2d-1de1-3b2d-8816-0ad90ee89903"
    with_mbid = {
        "uri": _URI.format("x.mp3"), "name": "n", "albumName": "a",
        "artistNames": "art", "albumArtistNames": "", "date": None,
        "track": None, "disc": None, "musicBrainzId": mbid,
    }
    assert song_id_from_cached_file(with_mbid) == song_uid_mbid(mbid)  # ums…

    no_mbid = {**with_mbid, "musicBrainzId": None}
    assert song_id_from_cached_file(no_mbid).startswith("uas")

    # Null album -> directory-name fallback ("Music") is used as the album.
    null_album = {**no_mbid, "albumName": None}
    from synth.uid import song_uid_auxio

    assert song_id_from_cached_file(null_album) == song_uid_auxio(
        name="n", album="Music", artists=["art"], album_artists=[]
    )


# --- optional: full-DB validation, only when both DBs are provided via env ----

_CACHE_ENV = os.environ.get("LJ_MUSIC_CACHE_DB")
_PLAYBACK_ENV = os.environ.get("LJ_PLAYBACK_DB")


@pytest.mark.skipif(
    not (_CACHE_ENV and _PLAYBACK_ENV and Path(_CACHE_ENV).exists() and Path(_PLAYBACK_ENV).exists()),
    reason="set LJ_MUSIC_CACHE_DB and LJ_PLAYBACK_DB to run the full-DB (814-row) validation",
)
def test_full_db_reproduces_all_ondevice_uids():  # pragma: no cover - opt-in extra
    with sqlite3.connect(f"file:{_PLAYBACK_ENV}?mode=ro", uri=True) as gt_con:
        ground_truth = {r[0] for r in gt_con.execute("SELECT songUid FROM TrackEmbeddingEntity")}
    with sqlite3.connect(f"file:{_CACHE_ENV}?mode=ro", uri=True) as cache_con:
        cache_con.row_factory = sqlite3.Row
        rows = [dict(r) for r in cache_con.execute("SELECT * FROM CachedFileData")]
    matched = sum(song_id_from_cached_file(r) in ground_truth for r in rows)
    rate = matched / len(rows)
    assert rate >= 0.99, f"only {matched}/{len(rows)} ({rate:.2%}) reproduced real ids"
