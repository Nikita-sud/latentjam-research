"""MagnaTagATune loader.

Expected on-disk layout (after ``scripts/download_mtat.py`` or manual download):

    data/raw/magnatagatune/
        annotations_final.csv  (or .tsv)
        mp3/
            0/0_chunk_1.mp3
            1/...
            ...

Each row in ``annotations_final.csv`` has 188 binary tag columns plus a
``mp3_path`` column. We reduce to the standard top-50 tags used in most
benchmarks; if you need the full 188, pass ``top_k_tags=188``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Standard top-50 MagnaTagATune tags — the canonical subset most papers use.
TOP_50_TAGS: tuple[str, ...] = (
    "guitar",
    "classical",
    "slow",
    "techno",
    "strings",
    "drums",
    "electronic",
    "rock",
    "fast",
    "piano",
    "ambient",
    "beat",
    "violin",
    "vocal",
    "synth",
    "female",
    "indian",
    "opera",
    "male",
    "singing",
    "vocals",
    "no vocals",
    "harpsichord",
    "loud",
    "quiet",
    "flute",
    "woman",
    "male vocal",
    "no vocal",
    "pop",
    "soft",
    "sitar",
    "solo",
    "man",
    "classic",
    "choir",
    "voice",
    "new age",
    "dance",
    "male voice",
    "female vocal",
    "beats",
    "harp",
    "cello",
    "no voice",
    "weird",
    "country",
    "metal",
    "female voice",
    "choral",
)


@dataclass
class MtatIndex:
    audio_root: Path
    annotations_root: Path
    annotations: pd.DataFrame  # index: mp3_path (str), columns: tag flags + 'clip_id'
    tag_names: tuple[str, ...]

    def tag_matrix_for(self, mp3_paths: list[str]) -> np.ndarray:
        """Return ``(N, T)`` 0/1 matrix for the given relative mp3 paths."""
        sub = self.annotations.reindex(mp3_paths)
        tags = sub[list(self.tag_names)].fillna(0).to_numpy(dtype=np.uint8)
        return tags


def _read_annotations(path: Path) -> pd.DataFrame:
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_mtat_index(
    audio_root: str | Path,
    annotations_root: str | Path,
    top_k_tags: int = 50,
) -> MtatIndex:
    audio_root = Path(audio_root)
    annotations_root = Path(annotations_root)

    candidates = [
        annotations_root / "annotations_final.csv",
        annotations_root / "annotations_final.tsv",
    ]
    csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            f"MagnaTagATune annotations not found in {annotations_root}. "
            f"Run `python scripts/download_mtat.py` first."
        )

    df = _read_annotations(csv_path)
    if "mp3_path" not in df.columns:
        raise ValueError(
            f"annotations file at {csv_path} missing 'mp3_path' column"
        )
    df = df.set_index("mp3_path")

    if top_k_tags == 50:
        tag_names = tuple(t for t in TOP_50_TAGS if t in df.columns)
    else:
        # All non-id columns.
        skip = {"clip_id"}
        tag_names = tuple(c for c in df.columns if c not in skip)[:top_k_tags]

    return MtatIndex(
        audio_root=audio_root,
        annotations_root=annotations_root,
        annotations=df,
        tag_names=tag_names,
    )


def align_store_to_mtat(store_df: pd.DataFrame, index: MtatIndex) -> pd.DataFrame:
    """Join store rows to MTAT annotations by relative mp3 path under the audio root."""
    df = store_df.copy()
    audio_root = index.audio_root.resolve()

    def _rel(p: str) -> str | None:
        try:
            return str(Path(p).resolve().relative_to(audio_root))
        except ValueError:
            return None

    df["mp3_path"] = df["path"].map(_rel)
    df = df.dropna(subset=["mp3_path"])
    df = df[df["mp3_path"].isin(index.annotations.index)]
    return df.reset_index(drop=True)
