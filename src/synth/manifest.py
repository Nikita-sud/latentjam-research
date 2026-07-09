"""Build the candidate-track manifest from the on-device caches, keyed by ``song_id``.

Reads ``CachedFileData`` rows from the phone's ``music_cache.db`` and computes
each track's canonical ``song_id`` (== on-device ``Music.UID``) via
:func:`synth.cachedfile.song_id_from_cached_file` -- the validated Task 1
bridge. That function (NOT a re-derivation of ``synth.uid.song_uid``) is the
only correct way to get from a cached row to a ``song_id``: it handles the
escaped ``;``-split for multi-value fields, the SAF-uri directory-name album
fallback, and ``album_artists``.

Audio features (``tempo``, ``energy``) are joined in from ``TrackEmbeddingEntity``
in the playback-persistence DB, keyed on ``songUid == song_id``. An optional
audited-tags CSV can backfill null ``genre``/``language`` by ``(title, artist)``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from synth.cachedfile import song_id_from_cached_file, split_multi

_COLUMNS = ["song_id", "title", "artist", "album", "genre", "year", "language", "bpm", "energy"]

# Separator used to build a composite (title, artist) lookup key. Chosen to be
# vanishingly unlikely to appear in real tag text (unlike "|" or ":").
_KEY_SEP = "␟"


def _read_table(db_path: str, table: str) -> list[dict[str, object]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute(f"SELECT * FROM {table}")]
    finally:
        con.close()


def _year(date: str | None) -> int | None:
    if not date:
        return None
    head = date[:4]
    return int(head) if len(head) == 4 and head.isdigit() else None


def _first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def _track_record(row: dict[str, object]) -> dict[str, object]:
    """One manifest row's metadata columns (everything except the joined features)."""
    bpm = row.get("bpm")
    return {
        "song_id": song_id_from_cached_file(row),
        "title": row.get("name") or None,
        "artist": _first_or_none(split_multi(row.get("artistNames"))),  # type: ignore[arg-type]
        "album": row.get("albumName") or None,
        "genre": _first_or_none(split_multi(row.get("genreNames"))),  # type: ignore[arg-type]
        "year": _year(row.get("date")),  # type: ignore[arg-type]
        "language": None,
        "bpm": float(bpm) if bpm is not None else None,
    }


def _load_features(playback_db: str) -> pd.DataFrame:
    rows = _read_table(playback_db, "TrackEmbeddingEntity")
    feats = pd.DataFrame(rows, columns=["songUid", "tempo", "energy"])
    return feats.rename(columns={"songUid": "song_id"})


def _lookup_key(titles: pd.Series, artists: pd.Series) -> pd.Series:
    return titles.fillna("").str.lower() + _KEY_SEP + artists.fillna("").str.lower()


def _backfill_from_audit(manifest: pd.DataFrame, audit_csv: str) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv, dtype=str)
    if not {"title", "artist"}.issubset(audit.columns):
        return manifest
    audit = audit.dropna(subset=["title", "artist"])
    audit = audit.assign(_key=_lookup_key(audit["title"], audit["artist"]))
    audit = audit.drop_duplicates(subset="_key", keep="first").set_index("_key")

    key = _lookup_key(manifest["title"], manifest["artist"])
    for col in ("genre", "language"):
        if col in audit.columns:
            backfilled = key.map(audit[col])
            manifest[col] = manifest[col].where(manifest[col].notna(), backfilled)
    return manifest


def build_manifest(
    music_cache_db: str, playback_db: str, audit_csv: str | None = None
) -> pd.DataFrame:
    """Build the per-track manifest keyed by ``song_id`` (Music.UID).

    Joins metadata from ``CachedFileData`` (``music_cache_db``) with audio
    features from ``TrackEmbeddingEntity`` (``playback_db``), and optionally
    backfills null ``genre``/``language`` from ``audit_csv``.
    """
    cache_rows = _read_table(music_cache_db, "CachedFileData")
    metadata_columns = [c for c in _COLUMNS if c != "energy"]
    manifest = pd.DataFrame(
        [_track_record(row) for row in cache_rows], columns=metadata_columns
    )

    feats = _load_features(playback_db)
    manifest = manifest.merge(feats, on="song_id", how="left")
    manifest["bpm"] = manifest["bpm"].where(manifest["bpm"].notna(), manifest["tempo"])
    manifest = manifest.drop(columns=["tempo"])

    if audit_csv and Path(audit_csv).exists():
        manifest = _backfill_from_audit(manifest, audit_csv)

    return manifest[_COLUMNS]
