"""Download the LAION-CLAP music checkpoint used as the distillation teacher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_URL = (
    "https://huggingface.co/lukewys/laion_clap/resolve/main/"
    "music_audioset_epoch_15_esc_90.14.pt"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
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
            with tqdm(total=total, initial=pos, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if not chunk:
                        continue
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("models/clap/music_audioset_epoch_15_esc_90.14.pt"),
    )
    args = ap.parse_args()
    if args.out.exists() and args.out.stat().st_size > 0:
        print(f"[skip] already present: {args.out}")
        return 0
    print(f"[download] {args.url}")
    download(args.url, args.out)
    print("Done. CLAP music checkpoint ready at", args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
