"""Write an EmbeddingStore from a trained student and cached target windows."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from predictor.store import EmbeddingStore, TrackRecord
from student.benchmark import _load_checkpoint
from student.config import (
    DEFAULT_FMA_TARGETS,
    DEFAULT_STUDENT_CKPT,
    STUDENT_MODEL_VERSION,
    MelConfig,
)
from student.data import DistillTargetDataset, collate_distill
from student.mel import LogMelExtractor


@click.command()
@click.option(
    "--targets",
    "targets_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(DEFAULT_FMA_TARGETS),
    show_default=True,
)
@click.option(
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(DEFAULT_STUDENT_CKPT),
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--split", default=None, help="Optional split filter.")
@click.option("--device", default="cpu", show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
def main(
    targets_path: Path,
    checkpoint_path: Path,
    out_path: Path,
    split: str | None,
    device: str,
    batch_size: int,
) -> None:
    df = pd.read_parquet(targets_path)
    if split:
        df = df[df["split"].astype(str) == split].reset_index(drop=True)
    df["teacher_embedding"] = df["teacher_embedding"].map(lambda x: np.asarray(x, dtype=np.float32))
    if df.empty:
        raise click.ClickException("no target rows matched")

    torch_device = torch.device(device)
    model = _load_checkpoint(checkpoint_path, torch_device)
    mel = LogMelExtractor(MelConfig()).to(torch_device).eval()
    loader = DataLoader(
        DistillTargetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_distill,
    )

    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for waveform, _target, _labels in loader:
            pred = model(mel(waveform.to(torch_device)))
            embeddings.extend(pred.cpu().numpy())

    work = df.copy()
    work["student_embedding"] = embeddings
    store = EmbeddingStore()
    for fma_track_id, group in work.groupby("fma_track_id", sort=True):
        matrix = np.stack(group["student_embedding"].to_list(), axis=0)
        emb = matrix.mean(axis=0)
        emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
        first = group.iloc[0]
        store.add(
            TrackRecord(
                track_id=f"fma-{int(fma_track_id):06d}",
                path=str(first["path"]),
                title=None,
                artist=None,
                album=None,
                genre=str(first["genre_top"]),
                year=None,
                embedding=emb.astype(np.float32),
                model_version=STUDENT_MODEL_VERSION,
            )
        )
    store.save(out_path)
    click.echo(f"Wrote {out_path} with {len(store)} tracks")


if __name__ == "__main__":
    main()
