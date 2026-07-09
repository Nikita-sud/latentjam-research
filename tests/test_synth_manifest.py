"""Tests for src/synth/manifest.py.

Task 1 extracted the validated CachedFileData -> song_id bridge into
``synth.cachedfile.song_id_from_cached_file``. The task-2 brief's illustrative
test computed ``song_id`` by calling ``synth.uid.song_uid(...)`` inline --
that is SUPERSEDED (it misses the escaped ``;``-split, the SAF-uri
directory-name album fallback, and ``album_artists``). This test instead
imports the same bridge function the implementation uses, so expectation and
implementation are guaranteed to agree, and exercises both the MBID path and
the null-album directory-fallback path.
"""

import csv
import sqlite3
import uuid

import pandas as pd

from synth.cachedfile import filename_stem, song_id_from_cached_file
from synth.manifest import build_manifest

_URI = (
    "content://com.android.externalstorage.documents/tree/primary%3AMusic/"
    "document/primary%3AMusic%2F{}"
)

_CACHE_COLUMNS = (
    "uri, name, artistNames, albumName, albumArtistNames, genreNames, date, "
    "bpm, durationMs, musicBrainzId, track, disc"
)


def _row(
    *,
    uri,
    name,
    artist_names,
    album_name,
    genre_names,
    date,
    bpm,
    mbid=None,
    track=None,
    disc=None,
    album_artist_names="",
):
    return {
        "uri": uri,
        "name": name,
        "artistNames": artist_names,
        "albumName": album_name,
        "albumArtistNames": album_artist_names,
        "genreNames": genre_names,
        "date": date,
        "bpm": bpm,
        "durationMs": 200_000,
        "musicBrainzId": mbid,
        "track": track,
        "disc": disc,
    }


def _make_cache_db(path, rows):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE CachedFileData (uri TEXT PRIMARY KEY, name TEXT, artistNames TEXT, "
        "albumName TEXT, albumArtistNames TEXT, genreNames TEXT, date TEXT, bpm INTEGER, "
        "durationMs INTEGER, musicBrainzId TEXT, track INTEGER, disc INTEGER)"
    )
    for r in rows:
        con.execute(
            f"INSERT INTO CachedFileData ({_CACHE_COLUMNS}) VALUES "
            f"(:uri, :name, :artistNames, :albumName, :albumArtistNames, :genreNames, :date, "
            f":bpm, :durationMs, :musicBrainzId, :track, :disc)",
            r,
        )
    con.commit()
    con.close()


def _make_playback_db(path, features):
    """features: dict[song_id] -> (tempo, energy)"""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE TrackEmbeddingEntity (songUid TEXT PRIMARY KEY, modelVersion TEXT, "
        "embedding BLOB, embeddedAtMs INTEGER, tempo REAL, energy REAL)"
    )
    for sid, (tempo, energy) in features.items():
        con.execute(
            "INSERT INTO TrackEmbeddingEntity VALUES (?,?,?,?,?,?)",
            (sid, "v3", b"\x00", 0, tempo, energy),
        )
    con.commit()
    con.close()


def _fixture_rows():
    """Three CachedFileData rows covering: plain auxio, MBID path, null-album dir-fallback."""
    row_auxio = _row(
        uri=_URI.format("For%20Leah.mp3"),
        name="For Leah",
        artist_names="Jake Chudnow",
        album_name="Singles",
        genre_names="Ambient",
        date="2011",
        bpm=90,
        track=1,
        disc=1,
    )
    mbid = str(uuid.uuid4())
    row_mbid = _row(
        uri=_URI.format("SongB.mp3"),
        name="Song B",
        artist_names="Artist B",
        album_name="Album B",
        genre_names="Pop",
        date="2020",
        bpm=120,
        mbid=mbid,
        track=2,
        disc=1,
    )
    # Null albumName -> directory-name fallback ("SubDir") is used to hash the id,
    # but the *manifest column* "album" stays None (only the id computation falls back).
    row_dirfallback = _row(
        uri=_URI.format("SubDir%2FTrackC.mp3"),
        name="Track C",
        artist_names="Artist C\\;Feat D",
        album_name=None,
        genre_names=None,
        date=None,
        bpm=None,
        track=None,
        disc=None,
    )
    return [row_auxio, row_mbid, row_dirfallback], mbid


def _expected_song_ids(rows):
    return [song_id_from_cached_file(r) for r in rows]


def test_build_manifest_joins_metadata_and_features(tmp_path):
    cache = tmp_path / "music_cache.db"
    playback = tmp_path / "playback.db"
    rows, _mbid = _fixture_rows()
    _make_cache_db(cache, rows)

    sid_auxio, sid_mbid, sid_dirfallback = _expected_song_ids(rows)
    _make_playback_db(
        playback,
        {
            sid_auxio: (90.0, 0.5),
            sid_mbid: (120.0, 0.8),
            sid_dirfallback: (100.0, 0.2),  # bpm tag is null -> should fall back to this tempo
        },
    )

    df = build_manifest(str(cache), str(playback))

    assert list(df.columns) == [
        "song_id",
        "title",
        "artist",
        "album",
        "genre",
        "year",
        "language",
        "bpm",
        "energy",
    ]
    assert set(df["song_id"]) == {sid_auxio, sid_mbid, sid_dirfallback}

    by_id = df.set_index("song_id")

    row_a = by_id.loc[sid_auxio]
    assert row_a["title"] == "For Leah"
    assert row_a["artist"] == "Jake Chudnow"
    assert row_a["album"] == "Singles"
    assert row_a["genre"] == "Ambient"
    assert int(row_a["year"]) == 2011
    assert row_a["bpm"] == 90.0
    assert row_a["energy"] == 0.5  # joined from playback db

    row_b = by_id.loc[sid_mbid]
    assert sid_mbid.startswith("ums")
    assert row_b["energy"] == 0.8
    assert row_b["bpm"] == 120.0

    row_c = by_id.loc[sid_dirfallback]
    assert row_c["album"] is None  # display column: no dir-name leakage, only the id used it
    assert row_c["artist"] == "Artist C;Feat D"  # escaped ';' unescaped by split_multi
    assert row_c["genre"] is None
    # "year" is an all-int-or-null column, so pandas promotes it to float64 (None -> NaN).
    assert pd.isna(row_c["year"])
    assert row_c["bpm"] == 100.0  # bpm tag null -> falls back to joined tempo
    assert row_c["energy"] == 0.2


def test_build_manifest_backfills_genre_and_language_from_audit_csv(tmp_path):
    cache = tmp_path / "music_cache.db"
    playback = tmp_path / "playback.db"
    rows, _mbid = _fixture_rows()
    _make_cache_db(cache, rows)
    sid_auxio, sid_mbid, sid_dirfallback = _expected_song_ids(rows)
    _make_playback_db(playback, {})

    audit_csv = tmp_path / "audit.csv"
    with open(audit_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["title", "artist", "genre", "language"])
        writer.writerow(["Track C", "Artist C;Feat D", "Downtempo", "instrumental"])

    df = build_manifest(str(cache), str(playback), audit_csv=str(audit_csv))
    by_id = df.set_index("song_id")

    row_c = by_id.loc[sid_dirfallback]
    assert row_c["genre"] == "Downtempo"
    assert row_c["language"] == "instrumental"

    # Row A already had a real genre tag -> audit backfill must not override it.
    row_a = by_id.loc[sid_auxio]
    assert row_a["genre"] == "Ambient"


def test_build_manifest_title_falls_back_to_filename_stem_when_name_is_null(tmp_path):
    # song_id_from_cached_file hashes the filename stem in for an untitled track
    # (null `name`); the manifest's displayed `title` must use that same stem,
    # not None, so the id and the human-readable title agree on what the track
    # is called (Fix 5).
    cache = tmp_path / "music_cache.db"
    playback = tmp_path / "playback.db"
    untitled_uri = _URI.format("UntitledTrack.mp3")
    row_untitled = _row(
        uri=untitled_uri,
        name=None,
        artist_names="Some Artist",
        album_name="Some Album",
        genre_names="Rock",
        date="2015",
        bpm=100,
        track=None,
        disc=None,
    )
    _make_cache_db(cache, [row_untitled])
    _make_playback_db(playback, {})

    df = build_manifest(str(cache), str(playback))

    expected_song_id = song_id_from_cached_file(row_untitled)
    expected_title = filename_stem(untitled_uri)
    assert expected_title == "UntitledTrack"

    assert len(df) == 1
    row = df.iloc[0]
    assert row["song_id"] == expected_song_id
    assert row["title"] == expected_title
