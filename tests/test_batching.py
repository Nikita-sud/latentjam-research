from __future__ import annotations

from pathlib import Path

import pytest

from encoder.batching import iterate_files


def _touch(p: Path, content: bytes = b"") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_iterate_files_picks_supported_extensions(tmp_path: Path) -> None:
    _touch(tmp_path / "a.flac")
    _touch(tmp_path / "sub" / "b.mp3")
    _touch(tmp_path / "sub" / "deeper" / "c.opus")
    _touch(tmp_path / "ignored.txt")
    _touch(tmp_path / "cover.jpg")

    found = sorted(p.name for p in iterate_files(tmp_path))
    assert found == ["a.flac", "b.mp3", "c.opus"]


def test_iterate_files_skips_hidden(tmp_path: Path) -> None:
    _touch(tmp_path / "a.flac")
    _touch(tmp_path / ".hidden" / "x.flac")
    _touch(tmp_path / "sub" / ".dot.flac")

    found = sorted(p.name for p in iterate_files(tmp_path))
    assert found == ["a.flac"]


def test_iterate_files_is_deterministic(tmp_path: Path) -> None:
    for n in ["c.flac", "a.flac", "b.flac"]:
        _touch(tmp_path / n)
    r1 = list(iterate_files(tmp_path))
    r2 = list(iterate_files(tmp_path))
    assert r1 == r2
    assert [p.name for p in r1] == ["a.flac", "b.flac", "c.flac"]


def test_iterate_files_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "a.flac"
    f.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        list(iterate_files(f))
