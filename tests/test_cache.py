from __future__ import annotations

from pathlib import Path

from encoder.cache import content_fingerprint


def test_fingerprint_is_stable_for_same_bytes(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    payload = b"\x01\x02\x03" * 4096
    a.write_bytes(payload)
    b.write_bytes(payload)
    assert content_fingerprint(a) == content_fingerprint(b)


def test_fingerprint_differs_when_size_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"\x00" * 100)
    b.write_bytes(b"\x00" * 101)
    assert content_fingerprint(a) != content_fingerprint(b)


def test_fingerprint_differs_when_head_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    pa = bytearray(b"\x00" * (3 << 20))
    pb = bytearray(pa)
    pb[0] = 0xFF  # change first byte
    a.write_bytes(bytes(pa))
    b.write_bytes(bytes(pb))
    assert content_fingerprint(a) != content_fingerprint(b)


def test_fingerprint_differs_when_tail_differs(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    pa = bytearray(b"\x00" * (3 << 20))
    pb = bytearray(pa)
    pb[-1] = 0xFF  # change last byte
    a.write_bytes(bytes(pa))
    b.write_bytes(bytes(pb))
    assert content_fingerprint(a) != content_fingerprint(b)


def test_fingerprint_full_hash_for_small_files(tmp_path: Path) -> None:
    # Files smaller than 2 * 1 MiB should be hashed in full; sanity check the
    # fast path works without IO errors.
    p = tmp_path / "small.bin"
    p.write_bytes(b"abc" * 17)
    assert len(content_fingerprint(p)) == 16
