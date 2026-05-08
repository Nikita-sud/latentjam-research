"""Benchmark a student checkpoint against cached LAION-CLAP teacher embeddings."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eval.metrics import recall_at_k_genre
from student.config import (
    DEFAULT_CLAP_MUSIC_CKPT,
    DEFAULT_FMA_TARGETS,
    DEFAULT_STUDENT_CKPT,
    STUDENT_MODEL_VERSION,
    MelConfig,
)
from student.data import DistillTargetDataset, collate_distill
from student.device import resolve_torch_device
from student.mel import LogMelExtractor
from student.metrics import cosine_summary, topk_overlap
from student.model import MelCnnStudent, count_parameters
from student.teacher import ClapTeacher, load_clap_window
from utils.wandb_log import log_metrics, log_summary, wandb_options, wandb_run


def _load_checkpoint(path: Path, device: torch.device) -> MelCnnStudent:
    ckpt = torch.load(path, map_location=device)
    embedding_dim = int(ckpt.get("model_config", {}).get("embedding_dim", 512))
    model = MelCnnStudent(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _read_targets(path: Path, split: str, limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["split"].astype(str) == split].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    df["teacher_embedding"] = df["teacher_embedding"].map(
        lambda x: np.asarray(x, dtype=np.float32)
    )
    if df.empty:
        raise click.ClickException(f"no rows found in {path} for split {split!r}")
    return df


@torch.inference_mode()
def _embed_student(
    model: MelCnnStudent,
    mel: LogMelExtractor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    student_batches: list[np.ndarray] = []
    teacher_batches: list[np.ndarray] = []
    labels: list[str] = []
    for waveform, target, batch_labels in loader:
        waveform = waveform.to(device)
        pred = model(mel(waveform))
        target = torch.nn.functional.normalize(target.to(device), p=2.0, dim=-1)
        student_batches.append(pred.cpu().numpy())
        teacher_batches.append(target.cpu().numpy())
        labels.extend(batch_labels)
    return (
        np.concatenate(student_batches, axis=0),
        np.concatenate(teacher_batches, axis=0),
        labels,
    )


@torch.inference_mode()
def _latency_ms(
    model: MelCnnStudent,
    mel: LogMelExtractor,
    loader: DataLoader,
    device: torch.device,
    *,
    n_iters: int,
) -> dict[str, float]:
    waveform, _target, _labels = next(iter(loader))
    waveform = waveform[:1].to(device)
    log_mel = mel(waveform)
    for _ in range(3):
        model(log_mel)
        model(mel(waveform))

    model_only: list[float] = []
    pipeline: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model(log_mel)
        model_only.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        model(mel(waveform))
        pipeline.append((time.perf_counter() - t0) * 1000.0)
    return {
        "student_model_p50_ms": float(np.percentile(model_only, 50)),
        "student_model_p95_ms": float(np.percentile(model_only, 95)),
        "student_pipeline_p50_ms": float(np.percentile(pipeline, 50)),
        "student_pipeline_p95_ms": float(np.percentile(pipeline, 95)),
    }


def _teacher_latency_ms(
    frame: pd.DataFrame,
    *,
    checkpoint: Path,
    device: str,
    amodel: str,
    n_iters: int,
) -> dict[str, float]:
    teacher = ClapTeacher(checkpoint=checkpoint, device=device, amodel=amodel)
    row = frame.iloc[0]
    waveform = load_clap_window(row["path"], int(row["start_sample"]))
    for _ in range(2):
        teacher.embed_waveforms([waveform])
    timings: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        teacher.embed_waveforms([waveform])
        timings.append((time.perf_counter() - t0) * 1000.0)
    return {
        "teacher_clap_p50_ms": float(np.percentile(timings, 50)),
        "teacher_clap_p95_ms": float(np.percentile(timings, 95)),
    }


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
@click.option("--split", default="test", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--latency-iters", type=int, default=50, show_default=True)
@click.option(
    "--teacher-checkpoint",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional LAION-CLAP checkpoint for teacher latency. "
        f"Typical path: {DEFAULT_CLAP_MUSIC_CKPT}"
    ),
)
@click.option("--teacher-amodel", default="HTSAT-base", show_default=True)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON report path.",
)
@wandb_options
def main(
    targets_path: Path,
    checkpoint_path: Path,
    split: str,
    device: str,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    latency_iters: int,
    teacher_checkpoint: Path | None,
    teacher_amodel: str,
    out: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...],
) -> None:
    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    df = _read_targets(targets_path, split, limit)
    model = _load_checkpoint(checkpoint_path, torch_device)
    mel = LogMelExtractor(MelConfig()).to(torch_device).eval()
    loader = DataLoader(
        DistillTargetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_distill,
    )

    student, teacher, labels = _embed_student(model, mel, loader, torch_device)
    k_values = tuple(k for k in (1, 5, 10, 20) if k < student.shape[0])
    report: dict[str, Any] = {
        "targets_path": str(targets_path),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "n": int(student.shape[0]),
        "model_version": STUDENT_MODEL_VERSION,
        "parameters": count_parameters(model),
        **cosine_summary(student, teacher),
        "topk_overlap": {str(k): v for k, v in topk_overlap(student, teacher, k_values=k_values).items()},
        "student_recall_at_k_genre": {
            str(k): v for k, v in recall_at_k_genre(student, labels, k_values).items()
        },
        "teacher_recall_at_k_genre": {
            str(k): v for k, v in recall_at_k_genre(teacher, labels, k_values).items()
        },
        **_latency_ms(model, mel, loader, torch_device, n_iters=latency_iters),
    }
    if teacher_checkpoint is not None:
        report.update(
            _teacher_latency_ms(
                df,
                checkpoint=teacher_checkpoint,
                device=str(torch_device),
                amodel=teacher_amodel,
                n_iters=latency_iters,
            )
        )

    payload = json.dumps(report, indent=2, sort_keys=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
    sys.stdout.write(payload + "\n")

    config = {
        "targets_path": str(targets_path),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "n": int(student.shape[0]),
        "requested_device": device,
        "device": str(torch_device),
        "batch_size": batch_size,
        "latency_iters": latency_iters,
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
        "teacher_amodel": teacher_amodel if teacher_checkpoint else None,
    }
    with wandb_run(
        wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=wandb_tags,
        config=config,
        job_type="student/benchmark",
    ) as run:
        loggable = {k: v for k, v in report.items() if k not in {"targets_path", "checkpoint"}}
        log_metrics(run, loggable)
        log_summary(run, loggable)


if __name__ == "__main__":
    main()
