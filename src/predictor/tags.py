"""Best-effort metadata extraction via mutagen.

Returns ``(title, artist, album, genre, year)`` with any field set to ``None``
when missing. Never raises — bad tags should not abort an embedding run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _first(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return _first(value[0])
    s = str(value).strip()
    return s or None


def _parse_year(value: Any) -> int | None:
    s = _first(value)
    if not s:
        return None
    head = s[:4]
    if head.isdigit():
        try:
            y = int(head)
            if 1900 <= y <= 2100:
                return y
        except ValueError:
            return None
    return None


def read_tags(
    path: str | Path,
) -> tuple[str | None, str | None, str | None, str | None, int | None]:
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        return None, None, None, None, None

    try:
        m = MutagenFile(str(path), easy=True)
    except Exception:
        return None, None, None, None, None
    if m is None or not getattr(m, "tags", None):
        return None, None, None, None, None

    tags = m.tags
    title = _first(tags.get("title"))
    artist = _first(tags.get("artist") or tags.get("albumartist"))
    album = _first(tags.get("album"))
    genre = _first(tags.get("genre"))
    year = _parse_year(tags.get("date") or tags.get("year") or tags.get("originaldate"))
    return title, artist, album, genre, year
