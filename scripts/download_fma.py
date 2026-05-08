"""Download and unpack FMA-small + FMA metadata.

Usage::

    python scripts/download_fma.py [--out data/raw]

Sizes:
    fma_metadata.zip   ~340 MB
    fma_small.zip     ~7.2 GB

URLs are documented at https://github.com/mdeff/fma; if a mirror dies, the
GitHub README has alternates. This script downloads with resume support and
validates SHA-1 checksums published in the FMA repo.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Checksums and URLs taken from the official FMA README (mdeff/fma).
ASSETS = [
    {
        "name": "fma_metadata.zip",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
        "sha1": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
        "extract_to": "fma_metadata",
    },
    {
        "name": "fma_small.zip",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "sha1": "ade154f733639d52e35e32f5593efe5be76c6d70",
        "extract_to": "fma_small",
    },
]


def _sha1(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    headers = {}
    mode = "wb"
    pos = 0
    if dest.exists():
        pos = dest.stat().st_size
        headers["Range"] = f"bytes={pos}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + pos
        with dest.open(mode) as f:
            with tqdm(
                total=total,
                initial=pos,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))


def _ensure_asset(asset: dict, raw_root: Path) -> Path:
    zip_path = raw_root / asset["name"]
    extract_root = raw_root / asset["extract_to"]

    if extract_root.exists() and any(extract_root.iterdir()):
        print(f"[skip] already extracted: {extract_root}")
        return extract_root

    if zip_path.exists() and _sha1(zip_path) == asset["sha1"]:
        print(f"[ok] cached zip matches checksum: {zip_path}")
    else:
        print(f"[download] {asset['url']}")
        _download(asset["url"], zip_path)
        actual = _sha1(zip_path)
        if actual != asset["sha1"]:
            raise RuntimeError(
                f"sha1 mismatch for {zip_path}: got {actual}, expected {asset['sha1']}"
            )

    print(f"[extract] {zip_path} -> {raw_root}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_root)
    return extract_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        _ensure_asset(asset, args.out)

    print("Done. FMA-small ready at", (args.out / "fma_small").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
