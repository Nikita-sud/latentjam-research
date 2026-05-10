"""Cache PANNs CNN14 teacher embeddings on FMA tracks for distillation.

PANNs CNN14 is an Apache-2.0-licensed audio tagging CNN pretrained on
AudioSet (~5000 hours). Unlike the LAION-CLAP-music checkpoint we used
earlier (CC-BY-NC-4.0), PANNs is shipping-friendly: nothing in this
training path is encumbered for the GPL-3 Android app.

Inputs:
- 32 kHz mono float32 audio. We re-decode each FMA window directly
  through ffmpeg with seek + duration so we never touch the cached
  24 kHz waveforms used for the SSL pipeline.

Outputs:
- A parquet at ``--out`` with one row per (track_id, window) and a
  2048-d ``teacher_embedding`` column. Same schema as
  ``prepare_clap_targets.py`` so ``train_distill.py`` picks it up
  unchanged (the student auto-derives ``embedding_dim`` from the
  parquet's first row, so a 2048-d teacher means a 2048-d student head).

Run:
    python scripts/prepare_panns_targets.py \\
        --audio-root /workspace/data/raw/fma_medium \\
        --metadata-root /workspace/data/raw/fma_metadata \\
        --subset medium \\
        --out /workspace/student_targets/fma_medium_panns_targets.parquet \\
        --device cuda \\
        --batch-size 64 \\
        --num-workers 8
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from student.data import iter_fma_windows, load_fma_manifest
from student.device import resolve_torch_device
from utils.wandb_log import log_artifact, log_metrics, log_summary, wandb_options, wandb_run

PANNS_SR = 32_000
PANNS_EMBEDDING_DIM = 2048
PANNS_MODEL_NAME = "panns-cnn14@audioset/v1"


def _decode_panns_window(args: tuple[str, float, float]) -> Optional[np.ndarray]:
    """Decode a 5-s window from ``path`` directly at 32 kHz mono float32.

    Uses ffmpeg subprocess (SIGBUS-safe — same approach as ssl_data._ffmpeg_decode).
    Seeks before -i so we don't waste time decoding the full 30-s clip and
    throwing 25 s away. ``start_seconds`` and ``duration_seconds`` are in
    real time, not 24 kHz samples.
    """
    path, start_seconds, duration_seconds = args
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel", "error",
                "-ss", f"{start_seconds:.3f}",
                "-i", path,
                "-t", f"{duration_seconds:.3f}",
                "-f", "f32le",
                "-ac", "1",
                "-ar", str(PANNS_SR),
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if not result.stdout:
        return None
    arr = np.frombuffer(result.stdout, dtype=np.float32).copy()
    expected = int(round(duration_seconds * PANNS_SR))
    if arr.size < expected:
        out = np.zeros(expected, dtype=np.float32)
        out[: arr.size] = arr
        arr = out
    elif arr.size > expected:
        arr = arr[:expected]
    return arr


def _split_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()}


@click.command()
@click.option(
    "--audio-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw/fma_medium"),
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
    default="medium",
    show_default=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("models/student/fma_medium_panns_targets.parquet"),
    show_default=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option("--windows-per-track", type=int, default=1, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="ffmpeg decode threads (each decodes one window at a time in a "
    "subprocess; subprocess.run releases the GIL while waiting).",
)
@click.option("--limit", type=int, default=None, help="Limit tracks before window expansion.")
@click.option(
    "--checkpoint",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to a Cnn14_mAP=0.431.pth checkpoint. If unset, "
    "panns_inference downloads it from zenodo on first call.",
)
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
    subset: str,
    out_path: Path,
    device: str,
    windows_per_track: int,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    checkpoint: Path | None,
    wandb_log_artifact: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...],
) -> None:
    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    manifest = load_fma_manifest(
        audio_root, metadata_root, subset=subset.lower(), existing_only=True
    )
    if limit is not None:
        manifest = manifest.head(limit)
    if manifest.empty:
        raise click.ClickException(
            f"no FMA audio rows under {audio_root}; check --subset / --audio-root"
        )

    print(f"manifest: {len(manifest)} tracks; subset={subset}; device={torch_device}")

    rows = list(iter_fma_windows(manifest, windows_per_track=windows_per_track))
    print(f"window plan: {len(rows)} windows ({windows_per_track} per track)")

    # Lazy import — keeps `--help` fast and isolates the heavy panns_inference dep.
    print("Loading PANNs CNN14...")
    from panns_inference import AudioTagging  # type: ignore[import-untyped]

    teacher_kwargs: dict[str, Any] = {"device": str(torch_device)}
    if checkpoint is not None:
        teacher_kwargs["checkpoint_path"] = str(checkpoint)
    teacher = AudioTagging(**teacher_kwargs)
    print("PANNs CNN14 loaded")

    decode_args = [
        (row["path"], float(row["start_seconds"]), float(row["duration_seconds"]))
        for row in rows
    ]

    output_rows: list[dict[str, Any]] = []
    skipped = 0
    pending_meta: list[dict[str, Any]] = []
    pending_wav: list[np.ndarray] = []

    def flush() -> None:
        nonlocal pending_meta, pending_wav
        if not pending_wav:
            return
        batch = np.stack(pending_wav, axis=0)
        with torch.no_grad():
            _, embeddings = teacher.inference(batch)
        for meta, emb in zip(pending_meta, embeddings, strict=True):
            output_rows.append({**meta, "teacher_embedding": emb.astype(np.float32)})
        pending_meta = []
        pending_wav = []

    # Decode windows in parallel via ThreadPoolExecutor. Each worker spawns
    # an ffmpeg subprocess; subprocess.run releases the GIL while waiting,
    # so threads scale across cores even with the GIL.
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        wav_iter = pool.map(_decode_panns_window, decode_args)
        for row, wav in tqdm(
            zip(rows, wav_iter, strict=True),
            total=len(rows),
            desc="panns targets",
            unit="window",
        ):
            if wav is None:
                skipped += 1
                continue
            pending_meta.append(row)
            pending_wav.append(wav)
            if len(pending_wav) >= batch_size:
                flush()
        flush()

    if skipped:
        print(f"warning: skipped {skipped} unreadable windows", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(output_rows)
    out_df["teacher_embedding"] = out_df["teacher_embedding"].map(lambda x: x.tolist())
    out_df.to_parquet(out_path, index=False)

    report = {
        "out_path": str(out_path),
        "audio_root": str(audio_root),
        "metadata_root": str(metadata_root),
        "subset": subset,
        "teacher_model": PANNS_MODEL_NAME,
        "teacher_embedding_dim": PANNS_EMBEDDING_DIM,
        "device": str(torch_device),
        "n_tracks": int(manifest["fma_track_id"].nunique()),
        "n_windows": int(len(out_df)),
        "n_skipped": int(skipped),
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
        job_type="student/prepare-panns-targets",
    ) as run:
        log_metrics(
            run,
            {
                "n_tracks": report["n_tracks"],
                "n_windows": report["n_windows"],
                "n_skipped": report["n_skipped"],
            },
        )
        log_summary(
            run,
            {
                "n_windows": report["n_windows"],
                "teacher_embedding_dim": PANNS_EMBEDDING_DIM,
            },
        )
        if wandb_log_artifact:
            log_artifact(
                run,
                str(out_path),
                name=out_path.stem,
                artifact_type="panns-targets",
                metadata=config,
            )


if __name__ == "__main__":
    main()
