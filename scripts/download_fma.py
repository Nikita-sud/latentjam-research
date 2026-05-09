"""Download and unpack FMA metadata + a chosen audio subset.

Usage::

    python scripts/download_fma.py [--subset small|medium|large|full] [--out data/raw]

Sizes (from mdeff/fma):

================  ===========  ===========  ====================
Subset            Tracks       ZIP size     Decoded @24kHz mono
================  ===========  ===========  ====================
small             8 000        7.2 GB       ~22 GB
medium            25 000       22 GB        ~75 GB
large             106 574      93 GB        ~300 GB
full (all songs)  106 574*     879 GB       ~7 TB
================  ===========  ===========  ====================

\\* full has the complete songs, not the 30-s clips. ``small`` / ``medium`` /
``large`` are nested supersets — large includes everything in medium, which
includes everything in small. The metadata zip is downloaded alongside any
subset.

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

# Checksums + URLs from the official FMA README (mdeff/fma).
METADATA_ASSET = {
    "name": "fma_metadata.zip",
    "url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
    "sha1": "f0df49ffe5f2a6008d7dc83c6915b31835dfe733",
    "extract_to": "fma_metadata",
}

SUBSET_ASSETS: dict[str, dict[str, str]] = {
    "small": {
        "name": "fma_small.zip",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "sha1": "ade154f733639d52e35e32f5593efe5be76c6d70",
        "extract_to": "fma_small",
        "approx_size": "7.2 GB",
    },
    "medium": {
        "name": "fma_medium.zip",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
        "sha1": "c67b69ea232021025fca9231fc1c7c1a063ab50b",
        "extract_to": "fma_medium",
        "approx_size": "22 GB",
    },
    "large": {
        "name": "fma_large.zip",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
        "sha1": "497109f4dd721066b5ce5e5f250ec604dc78939e",
        "extract_to": "fma_large",
        "approx_size": "93 GB",
    },
    "full": {
        "name": "fma_full.zip",
        "url": "https://os.unil.cloud.switch.ch/fma/fma_full.zip",
        "sha1": "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab",
        "extract_to": "fma_full",
        "approx_size": "879 GB",
    },
}


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--subset",
        choices=list(SUBSET_ASSETS.keys()),
        default="small",
        help="Which audio subset to fetch (default: small).",
    )
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the source .zip after extraction (default: delete to save disk).",
    )
    ap.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only download fma_metadata.zip; skip the audio subset.",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"target: {args.out.resolve()}")
    print(f"metadata: ~340 MB, audio subset {args.subset!r}: "
          f"{SUBSET_ASSETS[args.subset]['approx_size']}")

    _ensure_asset(METADATA_ASSET, args.out)

    if args.metadata_only:
        print("Done (metadata only).")
        return 0

    asset = SUBSET_ASSETS[args.subset]
    extract_root = _ensure_asset(asset, args.out)

    if not args.keep_zip:
        zip_path = args.out / asset["name"]
        if zip_path.exists():
            print(f"[cleanup] removing {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)")
            zip_path.unlink()

    print(f"Done. fma_{args.subset} ready at {extract_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
