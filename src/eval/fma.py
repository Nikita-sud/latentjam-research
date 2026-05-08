"""Free Music Archive (FMA) loader.

Expected on-disk layout (after ``scripts/download_fma.py`` or manual download):

    data/raw/fma_small/
        000/000002.mp3
        000/000005.mp3
        ...
    data/raw/fma_metadata/
        tracks.csv
        genres.csv
        ...

Track IDs are zero-padded 6-digit integers; the parent directory matches the
first three digits. The genre label we use is ``track.genre_top`` from
``tracks.csv`` — present and balanced across 8 classes for FMA-small.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class FmaIndex:
    """Path → FMA-track-id mapping plus a track-id → genre_top label."""

    audio_root: Path
    metadata_root: Path
    tracks: pd.DataFrame  # indexed by FMA track_id (int), columns include 'genre_top'

    @staticmethod
    def fma_track_id_from_path(path: str | Path) -> int | None:
        """``.../fma_small/000/000002.mp3`` -> 2."""
        stem = Path(path).stem
        if stem.isdigit() and len(stem) == 6:
            return int(stem)
        return None


def load_fma_index(
    audio_root: str | Path,
    metadata_root: str | Path,
    subset: str = "small",
) -> FmaIndex:
    """Load the FMA metadata for a subset (default: ``small``).

    The ``tracks.csv`` file has a 2-row header that pandas needs to be told
    about explicitly. We keep only ``track_id`` (index) and ``genre_top``.
    """
    audio_root = Path(audio_root)
    metadata_root = Path(metadata_root)

    csv_path = metadata_root / "tracks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"FMA tracks.csv not found at {csv_path}. "
            f"Run `python scripts/download_fma.py` first."
        )

    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    flat = pd.DataFrame(
        {
            "subset": tracks[("set", "subset")],
            "genre_top": tracks[("track", "genre_top")],
        }
    )
    flat.index.name = "fma_track_id"
    flat = flat[flat["subset"].isin(_subset_levels(subset))]
    flat = flat.dropna(subset=["genre_top"])
    return FmaIndex(audio_root=audio_root, metadata_root=metadata_root, tracks=flat)


def _subset_levels(subset: str) -> list[str]:
    s = subset.lower()
    if s == "small":
        return ["small"]
    if s == "medium":
        return ["small", "medium"]
    if s == "large":
        return ["small", "medium", "large"]
    raise ValueError(f"unknown subset {subset!r}")


def align_store_to_fma(store_df: pd.DataFrame, index: FmaIndex) -> pd.DataFrame:
    """Inner-join an embedding store on FMA track_id parsed from path.

    Returns the joined frame with columns ``track_id, path, embedding,
    fma_track_id, genre_top`` (one row per matched store track).
    """
    df = store_df.copy()
    df["fma_track_id"] = df["path"].map(FmaIndex.fma_track_id_from_path)
    df = df.dropna(subset=["fma_track_id"])
    df["fma_track_id"] = df["fma_track_id"].astype(int)
    joined = df.merge(
        index.tracks[["genre_top"]],
        left_on="fma_track_id",
        right_index=True,
        how="inner",
    )
    return joined
