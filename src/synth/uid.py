"""Reproduce musikr's Music.UID for songs so Python ids match on-device ids.

``UID.toString() == "u" + format.microNamespace + item.microNamespace + uuid``
(see ``latentjam/musikr/src/main/java/org/oxycblt/musikr/Music.kt``). AUXIO
songs hash their non-subjective metadata with SHA-256; the first 16 bytes of
the digest become the UUID. Java ``UUID(msb, lsb)`` built from the first 16
big-endian digest bytes is identical to ``uuid.UUID(bytes=digest[:16])``.

Field order and byte encoding are taken verbatim from the ``v363`` UID block in
``TagInterpreter.interpret`` and the ``MessageDigest.update`` extension
functions in ``musikr/.../util/LangUtil.kt``:

    update(songNameOrFileWithoutExtCorrect)  # String -> lowercase().toByteArray()
    update(albumNameOrDir)                   # String -> lowercase().toByteArray()
    update(date)                             # Date   -> toString().toByteArray()
    update(track)                            # Int    -> 4 little-endian bytes
    update(disc)                             # Int    -> 4 little-endian bytes
    update(artistNames)                      # List<String> -> each lowercased
    update(albumArtistNames)                 # List<String> -> each lowercased

Encoding rules (``LangUtil.update``):
- a non-null ``String`` is UTF-8 encoded after ``String.lowercase()``;
- a non-null ``Date`` is UTF-8 encoded from its ISO-8601 ``toString()`` WITHOUT
  lowercasing;
- a non-null ``Int`` is written as 4 little-endian bytes;
- a null ``String``/``Date``/``Int`` writes a single ``0x00`` byte;
- a ``List<String>`` hashes each element in order (empty list adds nothing).

Caller caveats:
- ``albumNameOrDir`` is NEVER null in the app -- a missing album tag falls back
  to the containing directory name. The caller must supply that fallback in
  ``album``; ``album=None`` reproduces the (never-taken in practice) null-string
  branch.
- ``date`` must already be the app's normalized ISO-8601 ``Date.toString()``
  form (e.g. ``"2022"``, ``"2015-06-15"``), not a raw tag string -- the app
  hashes the parsed/normalized value.
- The app also hashes ``albumArtistNames`` after ``artistNames``. This interface
  exposes only ``artists`` (album-artist names are assumed empty, the common
  case). Songs whose files carried a distinct album-artist tag will not
  reproduce through this function.
"""

from __future__ import annotations

import hashlib
import uuid

_AUXIO = "a"  # Format.AUXIO.microNamespace
_MUSICBRAINZ = "m"  # Format.MUSICBRAINZ.microNamespace
_SONG = "s"  # UID.Item.SONG.microNamespace

_NULL = b"\x00"


def _uuid_from_sha256(payload: bytes) -> uuid.UUID:
    return uuid.UUID(bytes=hashlib.sha256(payload).digest()[:16])


def _update_string(h: bytearray, string: str | None) -> None:
    """Mirror ``MessageDigest.update(string: String?)`` (lowercased UTF-8)."""
    if string is not None:
        h += string.lower().encode("utf-8")
    else:
        h += _NULL


def _update_int(h: bytearray, n: int | None) -> None:
    """Mirror ``MessageDigest.update(n: Int?)`` (4 little-endian bytes)."""
    if n is not None:
        h += (n & 0xFFFFFFFF).to_bytes(4, "little")
    else:
        h += _NULL


def _auxio_payload(
    *,
    name: str,
    album: str | None,
    artists: list[str],
    date: str | None,
    track: int | None,
    disc: int | None,
) -> bytes:
    """Assemble the exact bytes musikr hashes for a Song's auxio (v363) UID."""
    h = bytearray()
    _update_string(h, name)
    _update_string(h, album)
    # Date is hashed via its ISO-8601 toString(), NOT lowercased.
    if date is not None:
        h += date.encode("utf-8")
    else:
        h += _NULL
    _update_int(h, track)
    _update_int(h, disc)
    for artist in artists:  # artistNames, each lowercased
        _update_string(h, artist)
    return bytes(h)


def song_uid_auxio(
    *,
    name: str,
    album: str | None,
    artists: list[str],
    date: str | None = None,
    track: int | None = None,
    disc: int | None = None,
) -> str:
    """Return the ``uas…`` UID for a song that has no MusicBrainz recording id."""
    u = _uuid_from_sha256(
        _auxio_payload(
            name=name, album=album, artists=artists, date=date, track=track, disc=disc
        )
    )
    return f"u{_AUXIO}{_SONG}{u}"


def song_uid_mbid(mbid: str) -> str:
    """Return the MusicBrainz-format (``ums…``) UID for a recording MBID."""
    return f"u{_MUSICBRAINZ}{_SONG}{uuid.UUID(mbid)}"


def song_uid(
    *,
    mbid: str | None,
    name: str,
    album: str | None,
    artists: list[str],
    date: str | None = None,
    track: int | None = None,
    disc: int | None = None,
) -> str:
    """Reproduce ``Music.UID`` for a song: MBID path when present, else auxio hash."""
    if mbid:
        return song_uid_mbid(mbid)
    return song_uid_auxio(
        name=name, album=album, artists=artists, date=date, track=track, disc=disc
    )
