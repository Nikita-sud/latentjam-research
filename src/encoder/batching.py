"""Recursive file walker for audio libraries."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from encoder.audio_io import SUPPORTED_EXTS


def iterate_files(
    root: str | Path,
    exts: frozenset[str] | set[str] = SUPPORTED_EXTS,
) -> Iterator[Path]:
    """Yield audio file paths under ``root`` in deterministic order.

    Files are yielded in sorted (path) order so that repeated runs produce
    repeatable embedding stores. Hidden files and directories (leading dot)
    are skipped.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} is not a directory")

    exts_lower = {e.lower() for e in exts}

    for p in sorted(root_path.rglob("*")):
        if not p.is_file():
            continue
        if any(part.startswith(".") for part in p.relative_to(root_path).parts):
            continue
        if p.suffix.lower() in exts_lower:
            yield p
