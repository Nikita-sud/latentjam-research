"""Build an offline LAION-CLAP target cache for FMA distillation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from student.config import DEFAULT_CLAP_MUSIC_CKPT, DEFAULT_FMA_TARGETS
from student.data import iter_fma_windows, load_fma_manifest
from student.device import resolve_torch_device
from student.teacher import ClapTeacher, load_clap_window
from utils.wandb_log import log_artifact, log_metrics, log_summary, wandb_options, wandb_run


def _split_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()}


class ClapWindowDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], Any]:
        row = self.rows[idx]
        return row, load_clap_window(row["path"], int(row["start_sample"]))


def _collate_windows(batch: list[tuple[dict[str, Any], Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    rows, waveforms = zip(*batch, strict=True)
    return list(rows), list(waveforms)


@click.command()
@click.option(
    "--audio-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/fma_small"),
    show_default=True,
)
@click.option(
    "--metadata-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/fma_metadata"),
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_FMA_TARGETS),
    show_default=True,
)
@click.option(
    "--teacher-checkpoint",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_CLAP_MUSIC_CKPT),
    show_default=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option("--teacher-amodel", default="HTSAT-base", show_default=True)
@click.option("--windows-per-track", type=int, default=1, show_default=True)
@click.option("--batch-size", type=int, default=8, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--limit", type=int, default=None, help="Limit tracks before window expansion.")
@click.option(
    "--wandb-log-artifact",
    is_flag=True,
    default=False,
    help="Upload the target parquet as a W&B artifact.",
)
@wandb_options
def main(
    audio_root: Path,
    metadata_root: Path,
    out_path: Path,
    teacher_checkpoint: Path,
    device: str,
    teacher_amodel: str,
    windows_per_track: int,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    wandb_log_artifact: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...],
) -> None:
    manifest = load_fma_manifest(audio_root, metadata_root, subset="small", existing_only=True)
    if limit is not None:
        manifest = manifest.head(limit)
    if manifest.empty:
        raise click.ClickException(
            f"no FMA audio rows found under {audio_root}; run `make download-fma` first"
        )

    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    teacher = ClapTeacher(
        checkpoint=teacher_checkpoint,
        device=str(torch_device),
        amodel=teacher_amodel,
        enable_fusion=False,
    )

    rows = list(iter_fma_windows(manifest, windows_per_track=windows_per_track))
    output_rows: list[dict[str, Any]] = []

    loader = DataLoader(
        ClapWindowDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_windows,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    with tqdm(total=len(rows), desc="clap targets", unit="window") as progress:
        for batch_meta, batch_wav in loader:
            embeddings = teacher.embed_waveforms(batch_wav)
            for meta, emb in zip(batch_meta, embeddings, strict=True):
                output_rows.append({**meta, "teacher_embedding": emb.astype("float32")})
            progress.update(len(batch_meta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(output_rows)
    out_df["teacher_embedding"] = out_df["teacher_embedding"].map(lambda x: x.tolist())
    out_df.to_parquet(out_path, index=False)

    report = {
        "out_path": str(out_path),
        "audio_root": str(audio_root),
        "metadata_root": str(metadata_root),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_amodel": teacher_amodel,
        "requested_device": device,
        "device": str(torch_device),
        "n_tracks": int(manifest["fma_track_id"].nunique()),
        "n_windows": int(len(out_df)),
        "windows_per_track": int(windows_per_track),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "split_counts": _split_counts(out_df),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    sys.stdout.write(payload + "\n")

    config = {k: v for k, v in report.items() if k != "split_counts"}
    with wandb_run(
        wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=wandb_tags,
        config=config,
        job_type="student/prepare-clap-targets",
    ) as run:
        log_metrics(run, {"n_tracks": report["n_tracks"], "n_windows": report["n_windows"]})
        log_summary(run, {"n_windows": report["n_windows"]})
        if wandb_log_artifact:
            log_artifact(
                run,
                str(out_path),
                name=out_path.stem,
                artifact_type="clap-targets",
                metadata=config,
            )


if __name__ == "__main__":
    main()
