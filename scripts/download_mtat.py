"""Download and unpack MagnaTagATune (annotations + 3-part mp3 archive).

Usage::

    python scripts/download_mtat.py [--out data/raw/magnatagatune]

Sizes:
    annotations_final.csv  ~2 MB
    mp3.zip.001/.002/.003  ~3 GB combined

The mp3 archive is split into three parts that must be concatenated before
unzipping. URLs are documented at https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset.
If a mirror dies, the dataset has alternate hosting on Zenodo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ANNOTATIONS_URL = (
    "https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"
)

MP3_PARTS = [
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001",
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002",
    "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003",
]


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


def _join_parts(parts: list[Path], out: Path) -> None:
    with out.open("wb") as fout:
        for part in parts:
            with part.open("rb") as fin:
                shutil.copyfileobj(fin, fout, length=1 << 20)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/raw/magnatagatune"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    annotations = args.out / "annotations_final.csv"
    if annotations.exists() and annotations.stat().st_size > 0:
        print(f"[skip] {annotations} already present")
    else:
        print(f"[download] {ANNOTATIONS_URL}")
        _download(ANNOTATIONS_URL, annotations)

    mp3_root = args.out / "mp3"
    if mp3_root.exists() and any(mp3_root.iterdir()):
        print(f"[skip] {mp3_root} already extracted")
        return 0

    parts = []
    for url in MP3_PARTS:
        part_path = args.out / Path(url).name
        if not part_path.exists() or part_path.stat().st_size == 0:
            print(f"[download] {url}")
            _download(url, part_path)
        parts.append(part_path)

    full_zip = args.out / "mp3.zip"
    print(f"[concat] {len(parts)} parts -> {full_zip}")
    _join_parts(parts, full_zip)

    print(f"[extract] {full_zip} -> {mp3_root.parent}")
    with zipfile.ZipFile(full_zip) as z:
        z.extractall(args.out)

    print("Done. MagnaTagATune ready at", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
