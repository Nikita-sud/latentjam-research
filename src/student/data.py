"""Dataset helpers for FMA CLAP distillation."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from encoder.audio_io import load_audio
from student.config import TARGET_SR, WINDOW_SAMPLES, WINDOW_SECONDS


def fma_audio_path(audio_root: str | Path, fma_track_id: int) -> Path:
    root = Path(audio_root)
    return root / f"{fma_track_id // 1000:03d}" / f"{fma_track_id:06d}.mp3"


def load_fma_manifest(
    audio_root: str | Path = "data/raw/fma_small",
    metadata_root: str | Path = "data/raw/fma_metadata",
    *,
    subset: str = "small",
    existing_only: bool = True,
) -> pd.DataFrame:
    """Return FMA rows with audio path, genre, and official split."""
    audio_root = Path(audio_root)
    metadata_root = Path(metadata_root)
    csv_path = metadata_root / "tracks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"FMA tracks.csv not found at {csv_path}")

    tracks = pd.read_csv(csv_path, index_col=0, header=[0, 1])
    split_col = ("set", "split")
    split = tracks[split_col] if split_col in tracks.columns else "training"
    artist_id_col = ("artist", "id")
    artist_name_col = ("artist", "name")
    df = pd.DataFrame(
        {
            "fma_track_id": tracks.index.astype(int),
            "subset": tracks[("set", "subset")],
            "split": split,
            "genre_top": tracks[("track", "genre_top")],
            "artist_id": tracks[artist_id_col]
            if artist_id_col in tracks.columns
            else None,
            "artist_name": tracks[artist_name_col]
            if artist_name_col in tracks.columns
            else None,
        }
    )
    df = df[df["subset"].isin(_subset_levels(subset))]
    df = df.dropna(subset=["genre_top"])
    df["path"] = df["fma_track_id"].map(lambda tid: str(fma_audio_path(audio_root, int(tid))))
    if existing_only:
        # FMA-small ships a handful of truncated mp3s (< a few KB) that pass
        # exists() but fail libsndfile/libmpg123 parsing. Filter them out by
        # size so iter_fma_windows doesn't have to trip over them.
        def _is_real_audio(path: str) -> bool:
            p = Path(path)
            try:
                return p.is_file() and p.stat().st_size >= 50_000
            except OSError:
                return False

        df = df[df["path"].map(_is_real_audio)]
    return df.reset_index(drop=True)


def _subset_levels(subset: str) -> list[str]:
    s = subset.lower()
    if s == "small":
        return ["small"]
    if s == "medium":
        return ["small", "medium"]
    if s == "large":
        return ["small", "medium", "large"]
    raise ValueError(f"unknown FMA subset {subset!r}")


def audio_num_samples(path: str | Path, target_sr: int = TARGET_SR) -> int:
    """Estimate decoded sample count after resampling."""
    import soundfile as sf

    info = sf.info(str(path))
    return int(round((info.frames / float(info.samplerate)) * target_sr))


def window_starts(num_samples: int, windows_per_track: int, window_samples: int = WINDOW_SAMPLES) -> list[int]:
    if windows_per_track <= 0:
        raise ValueError("windows_per_track must be positive")
    if num_samples <= window_samples:
        return [0 for _ in range(windows_per_track)]
    max_start = num_samples - window_samples
    if windows_per_track == 1:
        return [max_start // 2]
    return [int(x) for x in np.linspace(0, max_start, windows_per_track)]


def iter_fma_windows(
    manifest: pd.DataFrame,
    *,
    windows_per_track: int,
    window_samples: int = WINDOW_SAMPLES,
) -> Iterator[dict[str, object]]:
    skipped = 0
    for row in manifest.itertuples(index=False):
        try:
            n_samples = audio_num_samples(row.path)
        except Exception as exc:
            # libsndfile/libmpg123 occasionally reject otherwise-real-looking
            # FMA mp3s (corrupted ID3, truncated payload, exotic codec parts).
            # Skip rather than aborting the whole cache build.
            skipped += 1
            print(
                f"[iter_fma_windows] skipping {row.path}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue
        for window_index, start in enumerate(
            window_starts(n_samples, windows_per_track, window_samples)
        ):
            yield {
                "fma_track_id": int(row.fma_track_id),
                "path": str(row.path),
                "genre_top": str(row.genre_top),
                "split": str(row.split),
                "window_index": int(window_index),
                "start_sample": int(start),
                "duration_samples": int(window_samples),
                "start_seconds": float(start) / TARGET_SR,
                "duration_seconds": WINDOW_SECONDS,
            }


def load_window(path: str | Path, start_sample: int, *, target_sr: int, window_samples: int) -> np.ndarray:
    wav = load_audio(path, target_sr=target_sr)
    start = max(0, int(start_sample))
    end = start + window_samples
    out = np.zeros(window_samples, dtype=np.float32)
    if start < wav.shape[0]:
        chunk = wav[start:end]
        out[: chunk.shape[0]] = chunk
    return out


class DistillTargetDataset(Dataset):
    """Rows from ``prepare_clap_targets`` yielding waveform windows and teacher vectors."""

    def __init__(self, frame: pd.DataFrame, *, cache_waveforms: bool = False):
        self.frame = frame.reset_index(drop=True)
        self._waveforms: list[torch.Tensor] | None = None
        if cache_waveforms:
            self._waveforms = [self._load_waveform(i) for i in range(len(self.frame))]

    def __len__(self) -> int:
        return len(self.frame)

    def _load_waveform(self, idx: int) -> torch.Tensor:
        row = self.frame.iloc[idx]
        wav = load_window(
            row["path"],
            int(row["start_sample"]),
            target_sr=TARGET_SR,
            window_samples=int(row["duration_samples"]),
        )
        return torch.from_numpy(wav)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[idx]
        wav = self._waveforms[idx] if self._waveforms is not None else self._load_waveform(idx)
        target = np.asarray(row["teacher_embedding"], dtype=np.float32)
        label = str(row["genre_top"])
        return wav, torch.from_numpy(target), label


def collate_distill(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    waveforms, targets, labels = zip(*batch, strict=True)
    return torch.stack(list(waveforms), dim=0), torch.stack(list(targets), dim=0), list(labels)
