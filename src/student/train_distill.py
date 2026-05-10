"""Train the mel-CNN student against cached LAION-CLAP embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from eval.metrics import recall_at_k_genre
from student.config import (
    DEFAULT_FMA_TARGETS,
    DEFAULT_STUDENT_CKPT,
    STUDENT_MODEL_VERSION,
    MelConfig,
)
from student.data import DistillTargetDataset, collate_distill
from student.device import resolve_torch_device
from student.mel import LogMelExtractor
from student.metrics import cosine_summary, relational_distillation_loss, topk_overlap
from student.model import MelCnnStudent, count_parameters
from utils.wandb_log import log_artifact, log_metrics, log_summary, wandb_options, wandb_run


def _read_targets(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "teacher_embedding" not in df.columns:
        raise ValueError(f"{path} missing teacher_embedding column")
    df["teacher_embedding"] = df["teacher_embedding"].map(
        lambda x: np.asarray(x, dtype=np.float32)
    )
    return df


def _normalize_matrix(values: list[np.ndarray]) -> np.ndarray:
    matrix = np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
    return matrix / norms


@torch.inference_mode()
def evaluate(
    model: MelCnnStudent,
    mel: LogMelExtractor,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    mel.eval()
    student_batches: list[np.ndarray] = []
    teacher_batches: list[np.ndarray] = []
    labels: list[str] = []
    losses: list[float] = []

    for batch_idx, (waveform, target, batch_labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        waveform = waveform.to(device)
        target = nn.functional.normalize(target.to(device), p=2.0, dim=-1)
        pred = model(mel(waveform))
        losses.append(float((1.0 - nn.functional.cosine_similarity(pred, target, dim=-1)).mean()))
        student_batches.append(pred.cpu().numpy())
        teacher_batches.append(target.cpu().numpy())
        labels.extend(batch_labels)

    if not student_batches:
        return {"loss": 0.0, "cosine_mean": 0.0}

    student = np.concatenate(student_batches, axis=0)
    teacher = np.concatenate(teacher_batches, axis=0)
    metrics: dict[str, Any] = {"loss": float(np.mean(losses)), **cosine_summary(student, teacher)}

    if len(labels) == student.shape[0] and len(set(labels)) > 1 and student.shape[0] > 2:
        k_values = tuple(k for k in (1, 5, 10, 20) if k < student.shape[0])
        metrics["student_recall_at_k_genre"] = {
            str(k): v for k, v in recall_at_k_genre(student, labels, k_values).items()
        }
        metrics["teacher_recall_at_k_genre"] = {
            str(k): v for k, v in recall_at_k_genre(teacher, labels, k_values).items()
        }
        metrics["topk_overlap"] = {
            str(k): v for k, v in topk_overlap(student, teacher, k_values=k_values).items()
        }
    return metrics


def _split_or_fallback(df: pd.DataFrame, split: str, fallback_frac: float = 0.1) -> pd.DataFrame:
    out = df[df["split"].astype(str) == split]
    if not out.empty:
        return out.reset_index(drop=True)
    n = max(1, int(round(len(df) * fallback_frac)))
    return df.tail(n).reset_index(drop=True)


@click.command()
@click.option(
    "--targets",
    "targets_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(DEFAULT_FMA_TARGETS),
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(DEFAULT_STUDENT_CKPT),
    show_default=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option("--epochs", type=int, default=5, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--relational-weight", type=float, default=0.25, show_default=True)
@click.option("--num-workers", type=int, default=0, show_default=True)
@click.option(
    "--cache-waveforms",
    is_flag=True,
    default=False,
    help="Decode waveform windows once into host RAM before training.",
)
@click.option("--train-split", default="training", show_default=True)
@click.option("--val-split", default="validation", show_default=True)
@click.option("--limit-train", type=int, default=None)
@click.option("--limit-val", type=int, default=None)
@click.option("--eval-max-batches", type=int, default=None)
@click.option(
    "--init-from",
    "init_from",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Warm-start the student from an existing SSL checkpoint (e.g. a "
    "``train_ssl.py`` output). Loads the backbone weights only; optimizer, "
    "scheduler, and projector start fresh. Pair with a small ``--lr`` "
    "(3e-5..1e-4) so the SSL features get refined toward the teacher rather "
    "than overwritten.",
)
@click.option(
    "--wandb-log-artifact",
    is_flag=True,
    default=False,
    help="Upload the best checkpoint as a W&B artifact.",
)
@wandb_options
def main(
    targets_path: Path,
    out_path: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    relational_weight: float,
    num_workers: int,
    cache_waveforms: bool,
    init_from: Path | None,
    train_split: str,
    val_split: str,
    limit_train: int | None,
    limit_val: int | None,
    eval_max_batches: int | None,
    wandb_log_artifact: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...],
) -> None:
    df = _read_targets(targets_path)
    train_df = df[df["split"].astype(str) == train_split].reset_index(drop=True)
    val_df = _split_or_fallback(df, val_split)
    if limit_train is not None:
        train_df = train_df.head(limit_train)
    if limit_val is not None:
        val_df = val_df.head(limit_val)
    if train_df.empty:
        raise click.ClickException(f"no training rows found for split {train_split!r}")
    if val_df.empty:
        raise click.ClickException(f"no validation rows found for split {val_split!r}")

    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    mel_config = MelConfig()
    mel = LogMelExtractor(mel_config).to(torch_device)
    model = MelCnnStudent(embedding_dim=len(train_df["teacher_embedding"].iloc[0])).to(torch_device)

    init_info: dict[str, Any] = {}
    if init_from is not None:
        ckpt = torch.load(str(init_from), map_location=torch_device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        init_info = {
            "init_from": str(init_from),
            "init_missing_keys": len(missing),
            "init_unexpected_keys": len(unexpected),
            "init_model_version": ckpt.get("model_version") if isinstance(ckpt, dict) else None,
        }
        print(
            f"warm-start from {init_from} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})",
            flush=True,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        DistillTargetDataset(train_df, cache_waveforms=cache_waveforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_distill,
    )
    val_loader = DataLoader(
        DistillTargetDataset(val_df, cache_waveforms=cache_waveforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_distill,
    )

    n_train_batches = max(1, len(train_loader))
    total_steps = max(1, epochs * n_train_batches)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=100.0,
    )

    config: dict[str, Any] = {
        "targets_path": str(targets_path),
        "out_path": str(out_path),
        "model_version": STUDENT_MODEL_VERSION,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "relational_weight": relational_weight,
        "cache_waveforms": cache_waveforms,
        **init_info,
        "requested_device": device,
        "device": str(torch_device),
        "mel": mel_config.to_dict(),
        "model": model.config_dict(),
    }

    best_cosine = -1.0
    best_report: dict[str, Any] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with wandb_run(
        wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=wandb_tags,
        config=config,
        job_type="student/train-distill",
    ) as run:
        for epoch in range(1, epochs + 1):
            model.train()
            mel.train()
            train_losses: list[float] = []
            train_cosines: list[float] = []
            for waveform, target, _labels in train_loader:
                waveform = waveform.to(torch_device)
                target = nn.functional.normalize(target.to(torch_device), p=2.0, dim=-1)
                pred = model(mel(waveform))
                cosine = nn.functional.cosine_similarity(pred, target, dim=-1)
                cosine_loss = (1.0 - cosine).mean()
                rel_loss = relational_distillation_loss(pred, target)
                loss = cosine_loss + relational_weight * rel_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()

                train_losses.append(float(loss.detach().cpu()))
                train_cosines.append(float(cosine.mean().detach().cpu()))

            val_metrics = evaluate(
                model,
                mel,
                val_loader,
                device=torch_device,
                max_batches=eval_max_batches,
            )
            epoch_metrics: dict[str, Any] = {
                "epoch": epoch,
                "train/loss": float(np.mean(train_losses)),
                "train/cosine_mean": float(np.mean(train_cosines)),
                "train/lr": float(scheduler.get_last_lr()[0]),
                **{f"val/{k}": v for k, v in val_metrics.items() if not isinstance(v, dict)},
            }
            for group in ("student_recall_at_k_genre", "teacher_recall_at_k_genre", "topk_overlap"):
                if group in val_metrics:
                    for k, v in val_metrics[group].items():
                        epoch_metrics[f"val/{group}/{k}"] = v
            log_metrics(run, epoch_metrics)

            val_cosine = float(val_metrics.get("cosine_mean", 0.0))
            if val_cosine > best_cosine:
                best_cosine = val_cosine
                best_report = {"epoch": epoch, **val_metrics}
                torch.save(
                    {
                        "model_version": STUDENT_MODEL_VERSION,
                        "model_state_dict": model.state_dict(),
                        "model_config": model.config_dict(),
                        "mel_config": mel_config.to_dict(),
                        "train_config": config,
                        "best_report": best_report,
                    },
                    out_path,
                )

        log_summary(run, {"best/cosine_mean": best_cosine, "model/parameters": count_parameters(model)})
        if wandb_log_artifact:
            log_artifact(
                run,
                str(out_path),
                name=out_path.stem,
                artifact_type="student-checkpoint",
                metadata={"best_cosine_mean": best_cosine, "model_version": STUDENT_MODEL_VERSION},
            )

    report = {
        "checkpoint": str(out_path),
        "model_version": STUDENT_MODEL_VERSION,
        "parameters": count_parameters(model),
        "best": best_report,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
