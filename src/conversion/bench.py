"""ORT CPU latency benchmark for an exported encoder.

This is a laptop proxy, not a mobile measurement. Mac M-series CPUs are
roughly 2-3x faster than a Pixel 6 CPU on transformer audio models — apply
that mentally; do not claim mobile numbers from these benches.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import onnxruntime as ort

from encoder.mert_encoder import MODEL_VERSION, WINDOW_SAMPLES
from utils.wandb_log import log_metrics, log_summary, wandb_options, wandb_run


@click.command()
@click.option(
    "--onnx",
    "onnx_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--n-iters", type=int, default=50, show_default=True)
@click.option("--n-warmup", type=int, default=5, show_default=True)
@click.option("--intra-op", type=int, default=4, show_default=True)
@click.option("--inter-op", type=int, default=1, show_default=True)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON report path.",
)
@wandb_options
def main(
    onnx_path: Path,
    n_iters: int,
    n_warmup: int,
    intra_op: int,
    inter_op: int,
    out: Path | None,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_name: str | None,
    wandb_tags: tuple[str, ...],
) -> None:
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = intra_op
    sess_opts.inter_op_num_threads = inter_op
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        onnx_path.as_posix(),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    rng = np.random.default_rng(0)
    waveform = rng.uniform(-0.5, 0.5, size=(1, WINDOW_SAMPLES)).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, {"waveform": waveform})

    timings_ms = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        t0 = time.perf_counter()
        session.run(None, {"waveform": waveform})
        timings_ms[i] = (time.perf_counter() - t0) * 1000.0

    report = {
        "onnx_path": str(onnx_path),
        "file_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "intra_op_threads": intra_op,
        "inter_op_threads": inter_op,
        "window_seconds": float(WINDOW_SAMPLES) / 24_000.0,
        "n_iters": int(n_iters),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p90_ms": float(np.percentile(timings_ms, 90)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "min_ms": float(timings_ms.min()),
        "mean_ms": float(timings_ms.mean()),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
    click.echo(payload)

    config = {
        "onnx_path": str(onnx_path),
        "model_version": MODEL_VERSION,
        "intra_op_threads": intra_op,
        "inter_op_threads": inter_op,
        "n_iters": int(n_iters),
        "n_warmup": int(n_warmup),
        "file_size_mb": report["file_size_mb"],
    }
    with wandb_run(
        wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        tags=wandb_tags,
        config=config,
        job_type="bench",
    ) as run:
        loggable = {k: v for k, v in report.items() if k != "onnx_path"}
        log_metrics(run, loggable)
        log_summary(run, loggable)


if __name__ == "__main__":
    main()
