"""Bridge musikr ``CachedFileData`` rows to canonical ``Music.UID`` song ids.

``CachedFileData`` (in ``music_cache.db``) stores the exact tags the app hashes
into a song's ``Music.UID``. This module reconstructs the hash inputs from a
cached row and returns ``song_uid(...)``:

- multi-value fields (``artistNames``/``albumArtistNames``) are the ``;``-joined,
  ``\\;``-escaped encoding of ``CacheDatabase.Converters.fromMultiValue``; we invert
  it with :func:`split_multi` (``ParseUtil.splitEscaped``);
- ``album`` follows the app's ``albumName ?: path.directory.name`` fallback, where
  the directory name is derived from the SAF document ``uri`` (:func:`directory_name`);
- a valid ``musicBrainzId`` selects the MBID path, else the metadata hash;
- ``date`` is already stored as ``Date.toString()`` (``CacheDatabase.fromDate``);
- ``name`` falls back to the file stem when the title tag is absent.

Task 2's ``build_manifest`` MUST call :func:`song_id_from_cached_file` rather than
re-deriving any of this.
"""

from __future__ import annotations

import urllib.parse
import uuid
from collections.abc import Mapping

from .uid import song_uid


def split_multi(value: str | None) -> list[str]:
    """Invert ``CacheDatabase.Converters.fromMultiValue`` (``ParseUtil.splitEscaped``).

    Values are ``;``-joined with any literal ``;`` escaped as ``\\;``. Split on
    unescaped ``;`` and unescape ``\\;`` -> ``;`` to recover the exact list the app
    hashed. ``None``/empty -> ``[]``.
    """
    if not value:
        return []
    out: list[str] = []
    cur: list[str] = []
    i, n = 0, len(value)
    while i < n:
        a = value[i]
        b = value[i + 1] if i + 1 < n else None
        if a == ";":
            out.append("".join(cur))
            cur = []
            i += 1
        elif b == ";" and a == "\\":
            cur.append(b)
            i += 2
        else:
            cur.append(a)
            i += 1
    if cur:
        out.append("".join(cur))
    return out


def _document_relpath(uri: str) -> list[str]:
    """Path components of a SAF document ``uri`` relative to its volume.

    Mirrors ``DocumentPathFactory.fromDocumentId`` + ``Components.parseUnix``:
    percent-decode the id after ``/document/``, split off the volume on ``:``
    (``File.pathSeparator``), then split the remainder on ``/`` dropping empties.
    ``content://…/document/primary%3AMusic%2Ffile.mp3`` -> ``["Music", "file.mp3"]``.
    """
    marker = "/document/"
    idx = uri.find(marker)
    doc = urllib.parse.unquote(uri[idx + len(marker):] if idx >= 0 else uri)
    rel = doc.split(":", 1)
    if len(rel) < 2:
        return []
    return [c for c in rel[1].split("/") if c]


def directory_name(uri: str) -> str | None:
    """``song.file.path.directory.name`` -- the file's parent directory component."""
    parent = _document_relpath(uri)[:-1]
    return parent[-1] if parent else None


def _filename_stem(uri: str) -> str:
    """``filename.substringBeforeLast('.')`` -- name fallback when the title tag is absent."""
    comps = _document_relpath(uri)
    last = comps[-1] if comps else ""
    return last.rsplit(".", 1)[0] if "." in last else last


def valid_mbid(value: str | None) -> str | None:
    """Return ``value`` if it parses as a UUID (``toUuidOrNull``), else ``None``."""
    if not value:
        return None
    try:
        uuid.UUID(value)
    except ValueError:
        return None
    return value


def song_id_from_cached_file(row: Mapping[str, object]) -> str:
    """Compute the canonical ``Music.UID`` for a ``CachedFileData`` row.

    ``row`` is a mapping with keys ``uri, name, artistNames, albumName,
    albumArtistNames, date, track, disc, musicBrainzId`` (any may be missing/None).
    """
    uri = str(row.get("uri") or "")
    name = row.get("name")
    if not name:  # missing title tag -> file stem (substringBeforeLast('.'))
        name = _filename_stem(uri)
    album_name = row.get("albumName")
    album = album_name if album_name is not None else directory_name(uri)
    track = row.get("track")
    disc = row.get("disc")
    return song_uid(
        mbid=valid_mbid(row.get("musicBrainzId")),  # type: ignore[arg-type]
        name=str(name),
        album=album,  # type: ignore[arg-type]
        artists=split_multi(row.get("artistNames")),  # type: ignore[arg-type]
        album_artists=split_multi(row.get("albumArtistNames")),  # type: ignore[arg-type]
        date=(str(row.get("date")) if row.get("date") else None),
        track=int(track) if track is not None else None,
        disc=int(disc) if disc is not None else None,
    )
