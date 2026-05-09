"""Self-supervised VICReg training for the mel-CNN student on FMA-small.

No teacher, no CLAP, no pre-cached targets. Decodes all FMA-small clips into
RAM once (parallel via ThreadPoolExecutor), then samples two random 5-s crops
per item for VICReg's invariance / variance / covariance objective.

Auto-selects MPS if available (M-series Mac) via ``resolve_torch_device``;
falls back cleanly to CUDA or CPU. Wandb logging is opt-in via the standard
``--wandb-*`` flag set; per-epoch metrics include the three loss components,
the view-pair cosine, learning rate, and val-split genre recall@k.
"""

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
from student.augment import SpecAugment, random_gain, random_polarity
from student.config import MelConfig
from student.data import load_fma_manifest
from student.device import resolve_torch_device
from student.mel import LogMelExtractor
from student.model import MelCnnStudent, count_parameters
from student.projector import VICRegProjector
from student.ssl_data import FmaSslDataset, cache_or_decode_clips, collate_ssl
from student.ssl_loss import VICRegLoss
from utils.wandb_log import log_metrics, log_summary, wandb_options, wandb_run

SSL_MODEL_VERSION = "mel-cnn-vicreg@96mel-5s/512d/v1"


@torch.inference_mode()
def evaluate(
    student: MelCnnStudent,
    mel: LogMelExtractor,
    projector: VICRegProjector,
    vicreg: VICRegLoss,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    student.eval()
    mel.eval()
    projector.eval()
    embeds: list[np.ndarray] = []
    labels: list[str] = []
    view_cosines: list[float] = []
    losses: list[float] = []
    sims: list[float] = []
    stds: list[float] = []
    covs: list[float] = []

    for v1, v2, lbls in val_loader:
        v1 = v1.to(device)
        v2 = v2.to(device)
        # No augmentation on val: deterministic, fair-comparison eval.
        m1 = mel(v1)
        m2 = mel(v2)
        emb1 = student(m1)
        emb2 = student(m2)
        if emb1.shape[0] >= 2:
            z1 = projector(emb1)
            z2 = projector(emb2)
            loss, comps = vicreg(z1, z2)
            losses.append(float(loss.detach().cpu()))
            sims.append(comps["sim"])
            stds.append(comps["std"])
            covs.append(comps["cov"])
        cos = nn.functional.cosine_similarity(emb1, emb2, dim=-1)
        view_cosines.extend(cos.cpu().tolist())
        embeds.append(emb1.cpu().numpy())
        labels.extend(lbls)

    if not embeds:
        return {}
    matrix = np.concatenate(embeds, axis=0)
    metrics: dict[str, Any] = {
        "view_cosine_mean": float(np.mean(view_cosines)) if view_cosines else 0.0,
    }
    if losses:
        metrics["loss"] = float(np.mean(losses))
        metrics["loss_sim"] = float(np.mean(sims))
        metrics["loss_std"] = float(np.mean(stds))
        metrics["loss_cov"] = float(np.mean(covs))
    unique_labels = sorted(set(labels))
    if len(unique_labels) > 1 and len(labels) > 5:
        k_values = tuple(k for k in (1, 5, 10, 20) if k < len(labels))
        recall = recall_at_k_genre(matrix, labels, k_values)
        for k, v in recall.items():
            metrics[f"recall_at_k_genre/{k}"] = float(v)
    return metrics


def _prepare_splits(
    manifest: pd.DataFrame, val_fraction: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = manifest[manifest["split"] == "training"].reset_index(drop=True)
    val = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    if val.empty:
        n_val = max(1, int(round(len(manifest) * val_fraction)))
        val = manifest.tail(n_val).reset_index(drop=True)
        train = manifest.head(len(manifest) - n_val).reset_index(drop=True)
    return train, val


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
    "--subset",
    type=click.Choice(["small", "medium", "large"], case_sensitive=False),
    default="small",
    show_default=True,
    help="FMA subset filter for the manifest. 'small' keeps only the 8k "
    "balanced-genre tracks; 'medium' keeps small+medium (~25k); 'large' "
    "keeps everything (~106k). Audio-root must point at the matching "
    "extracted directory (e.g. data/raw/fma_large for --subset large).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("models/student/mel_cnn_ssl.pt"),
    show_default=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option("--epochs", type=int, default=30, show_default=True)
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--decode-workers", type=int, default=12, show_default=True)
@click.option(
    "--data-workers",
    type=int,
    default=4,
    show_default=True,
    help="Per-DataLoader subprocess count for the train + val loaders. "
    "0 = main-thread loading (slow on GPU due to forward/backward stalls "
    "while augmenting + collating); 4-8 saturates an A100 with bs=512.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models/cache/ssl_audio"),
    show_default=True,
    help="Directory for the on-disk decoded-waveform cache. Set to '' to disable.",
)
@click.option("--projector-dim", type=int, default=2048, show_default=True)
@click.option("--projector-hidden", type=int, default=2048, show_default=True)
@click.option("--sim-coef", type=float, default=25.0, show_default=True)
@click.option("--std-coef", type=float, default=25.0, show_default=True)
@click.option("--cov-coef", type=float, default=1.0, show_default=True)
@click.option("--gain-jitter-db", type=float, default=3.0, show_default=True)
@click.option("--polarity-flip-prob", type=float, default=0.5, show_default=True)
@click.option(
    "--artist-positive-prob",
    type=float,
    default=0.0,
    show_default=True,
    help="Probability of pairing view2 with a different track from the same artist (train only).",
)
@click.option("--freq-masks", type=int, default=2, show_default=True)
@click.option("--time-masks", type=int, default=2, show_default=True)
@click.option("--limit-tracks", type=int, default=None)
@click.option("--seed", type=int, default=0, show_default=True)
@wandb_options
def main(
    audio_root: Path,
    metadata_root: Path,
    subset: str,
    out_path: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    decode_workers: int,
    data_workers: int,
    cache_dir: Path | None,
    projector_dim: int,
    projector_hidden: int,
    sim_coef: float,
    std_coef: float,
    cov_coef: float,
    gain_jitter_db: float,
    polarity_flip_prob: float,
    artist_positive_prob: float,
    freq_masks: int,
    time_masks: int,
    limit_tracks: int | None,
    seed: int,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...],
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    manifest = load_fma_manifest(
        audio_root, metadata_root, subset=subset.lower(), existing_only=True
    )
    if limit_tracks is not None:
        manifest = manifest.head(limit_tracks).reset_index(drop=True)
    train_mfst, val_mfst = _prepare_splits(manifest)
    if train_mfst.empty:
        raise click.ClickException("no training tracks found in manifest")
    if val_mfst.empty:
        raise click.ClickException("no validation tracks found in manifest")

    click.echo(
        f"manifest: {len(manifest)} tracks; train={len(train_mfst)}, val={len(val_mfst)}"
    )

    cache_target = cache_dir if cache_dir and str(cache_dir) else None
    train_clips = cache_or_decode_clips(
        train_mfst,
        cache_dir=cache_target,
        cache_label="fma_train",
        num_workers=decode_workers,
        desc="decode train",
    )
    val_clips = cache_or_decode_clips(
        val_mfst,
        cache_dir=cache_target,
        cache_label="fma_val",
        num_workers=decode_workers,
        desc="decode val",
    )

    train_ds = FmaSslDataset(
        train_mfst,
        train_clips,
        seed=seed,
        artist_positive_prob=artist_positive_prob,
    )
    val_ds = FmaSslDataset(
        val_mfst,
        val_clips,
        seed=seed + 1,
        artist_positive_prob=0.0,
    )
    click.echo(
        f"valid samples: train={len(train_ds)}, val={len(val_ds)}; "
        f"artist-positive pool={train_ds.artist_pair_pool_size}"
    )
    if len(train_ds) == 0:
        raise click.ClickException("no valid train samples after decoding")
    if len(val_ds) == 0:
        raise click.ClickException("no valid val samples after decoding")

    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    pin_memory = torch_device.type == "cuda"
    persistent = data_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_workers,
        collate_fn=collate_ssl,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(batch_size, max(2, len(val_ds))),
        shuffle=False,
        num_workers=data_workers,
        collate_fn=collate_ssl,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    mel_config = MelConfig()
    mel = LogMelExtractor(mel_config).to(torch_device)
    student = MelCnnStudent().to(torch_device)
    projector = VICRegProjector(
        in_dim=student.embedding_dim,
        hidden_dim=projector_hidden,
        out_dim=projector_dim,
    ).to(torch_device)
    spec_aug = SpecAugment(
        freq_masks=freq_masks, time_masks=time_masks
    ).to(torch_device)
    vicreg = VICRegLoss(
        sim_coef=sim_coef, std_coef=std_coef, cov_coef=cov_coef
    )

    params = list(student.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
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
        "model_version": SSL_MODEL_VERSION,
        "n_train_tracks": int(len(train_mfst)),
        "n_val_tracks": int(len(val_mfst)),
        "n_train_samples": int(len(train_ds)),
        "n_val_samples": int(len(val_ds)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "projector_dim": int(projector_dim),
        "projector_hidden": int(projector_hidden),
        "sim_coef": float(sim_coef),
        "std_coef": float(std_coef),
        "cov_coef": float(cov_coef),
        "gain_jitter_db": float(gain_jitter_db),
        "polarity_flip_prob": float(polarity_flip_prob),
        "artist_positive_prob": float(artist_positive_prob),
        "artist_pair_pool": int(train_ds.artist_pair_pool_size),
        "freq_masks": int(freq_masks),
        "time_masks": int(time_masks),
        "decode_workers": int(decode_workers),
        "requested_device": device,
        "device": str(torch_device),
        "model": student.config_dict(),
        "projector": projector.config_dict(),
        "mel": mel_config.to_dict(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_recall_10 = -1.0
    best_report: dict[str, Any] = {}

    with wandb_run(
        wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=wandb_tags,
        config=config,
        job_type="student/train-ssl",
    ) as run:
        for epoch in range(1, epochs + 1):
            student.train()
            mel.train()
            spec_aug.train()
            projector.train()

            train_losses: list[float] = []
            sim_terms: list[float] = []
            std_terms: list[float] = []
            cov_terms: list[float] = []
            train_view_cosines: list[float] = []

            for v1, v2, _labels in train_loader:
                v1 = v1.to(torch_device, non_blocking=True)
                v2 = v2.to(torch_device, non_blocking=True)

                v1 = random_gain(v1, max_db=gain_jitter_db)
                v2 = random_gain(v2, max_db=gain_jitter_db)
                v1 = random_polarity(v1, p=polarity_flip_prob)
                v2 = random_polarity(v2, p=polarity_flip_prob)

                m1 = spec_aug(mel(v1))
                m2 = spec_aug(mel(v2))

                emb1 = student(m1)
                emb2 = student(m2)
                z1 = projector(emb1)
                z2 = projector(emb2)

                loss, comps = vicreg(z1, z2)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()

                train_losses.append(float(loss.detach().cpu()))
                sim_terms.append(comps["sim"])
                std_terms.append(comps["std"])
                cov_terms.append(comps["cov"])
                cos = nn.functional.cosine_similarity(
                    emb1.detach(), emb2.detach(), dim=-1
                ).mean()
                train_view_cosines.append(float(cos.cpu()))

            val_metrics = evaluate(student, mel, projector, vicreg, val_loader, torch_device)

            epoch_metrics: dict[str, Any] = {
                "epoch": epoch,
                "train/loss": float(np.mean(train_losses)),
                "train/loss_sim": float(np.mean(sim_terms)),
                "train/loss_std": float(np.mean(std_terms)),
                "train/loss_cov": float(np.mean(cov_terms)),
                "train/view_cosine_mean": float(np.mean(train_view_cosines)),
                "train/lr": float(scheduler.get_last_lr()[0]),
            }
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[f"val/{k}"] = float(v)
            log_metrics(run, epoch_metrics)

            recall_10 = float(val_metrics.get("recall_at_k_genre/10", 0.0))
            if recall_10 > best_recall_10:
                best_recall_10 = recall_10
                best_report = {"epoch": epoch, **val_metrics}
                torch.save(
                    {
                        "model_version": SSL_MODEL_VERSION,
                        "model_state_dict": student.state_dict(),
                        "projector_state_dict": projector.state_dict(),
                        "model_config": student.config_dict(),
                        "projector_config": projector.config_dict(),
                        "mel_config": mel_config.to_dict(),
                        "train_config": config,
                        "best_report": best_report,
                    },
                    out_path,
                )

        log_summary(
            run,
            {
                "best/val_recall_at_10": best_recall_10,
                "model/parameters": count_parameters(student),
                "projector/parameters": count_parameters(projector),
            },
        )

    report = {
        "checkpoint": str(out_path),
        "model_version": SSL_MODEL_VERSION,
        "parameters": count_parameters(student),
        "projector_parameters": count_parameters(projector),
        "best": best_report,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
