"""Content fingerprint for audio files, used as the stable ``track_id``.

We hash a sparse view of the file (size + first/last 1 MiB) rather than the full
contents because audio collections are big and we re-scan often. Stable across
path renames and library moves; collisions are negligible for libraries up to
~10^9 files (xxhash64 has 2^32 expected collisions at that scale, well above any
realistic music collection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import xxhash

_HEAD_TAIL_BYTES: Final[int] = 1 << 20  # 1 MiB


def content_fingerprint(path: str | Path, head_tail_bytes: int = _HEAD_TAIL_BYTES) -> str:
    """Return a 16-hex-char xxhash64 over (size, head bytes, tail bytes).

    Files smaller than ``2 * head_tail_bytes`` are hashed in full.
    """
    p = Path(path)
    size = p.stat().st_size

    h = xxhash.xxh64()
    h.update(size.to_bytes(8, "little", signed=False))

    with p.open("rb") as f:
        if size <= 2 * head_tail_bytes:
            h.update(f.read())
        else:
            h.update(f.read(head_tail_bytes))
            f.seek(size - head_tail_bytes)
            h.update(f.read(head_tail_bytes))

    return h.hexdigest()
