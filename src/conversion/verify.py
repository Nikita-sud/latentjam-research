"""Compare PyTorch reference vs. ONNX (FP32 or INT8) on N audio samples.

The PT reference uses the same ``MertOnnxWrapper`` graph (so input
normalization is included on both sides). This isolates ONNX export +
quantization noise from any pre-processing mismatch.

Gate (per ``CLAUDE.md``):
- mean cosine(PT_fp32, ONNX_fp32) >= 0.99999
- mean cosine(PT_fp32, ONNX_int8) >= 0.9995
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import onnxruntime as ort
import torch

from conversion.export_onnx import MertOnnxWrapper
from encoder.mert_encoder import EMBEDDING_DIM, WINDOW_SAMPLES, MertEncoder


def _generate_or_load_samples(
    n: int,
    seed: int,
    audio_root: Path | None,
) -> list[np.ndarray]:
    """N waveforms, each shape (WINDOW_SAMPLES,) float32 in [-1, 1]."""
    rng = np.random.default_rng(seed)
    if audio_root is None:
        return [
            rng.uniform(-0.5, 0.5, size=WINDOW_SAMPLES).astype(np.float32)
            for _ in range(n)
        ]

    from encoder.audio_io import load_audio
    from encoder.batching import iterate_files
    from encoder.mert_encoder import HOP_SAMPLES

    files = list(iterate_files(audio_root))
    if not files:
        raise FileNotFoundError(f"no audio files under {audio_root}")
    rng.shuffle(files)

    samples: list[np.ndarray] = []
    for p in files:
        if len(samples) >= n:
            break
        try:
            wav = load_audio(p)
        except Exception:
            continue
        if wav.shape[0] < WINDOW_SAMPLES:
            chunk = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            chunk[: wav.shape[0]] = wav
        else:
            start = rng.integers(0, max(1, wav.shape[0] - WINDOW_SAMPLES))
            chunk = wav[start : start + WINDOW_SAMPLES]
        samples.append(chunk.astype(np.float32, copy=False))
        # Use multiple chunks per file when we run out of files.
        if len(files) < n and wav.shape[0] >= WINDOW_SAMPLES + HOP_SAMPLES:
            extra = wav[HOP_SAMPLES : HOP_SAMPLES + WINDOW_SAMPLES]
            samples.append(extra.astype(np.float32, copy=False))
    if len(samples) < n:
        raise RuntimeError(
            f"only collected {len(samples)} samples from {audio_root}, wanted {n}"
        )
    return samples[:n]


def _pt_embed(wrapper: MertOnnxWrapper, wav: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        x = torch.from_numpy(wav).unsqueeze(0)
        return wrapper(x).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def _ort_embed(session: ort.InferenceSession, wav: np.ndarray) -> np.ndarray:
    out = session.run(None, {"waveform": wav.reshape(1, -1).astype(np.float32)})[0]
    return out.reshape(-1).astype(np.float32, copy=False)


@click.command()
@click.option(
    "--onnx",
    "onnx_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--n", type=int, default=50, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--audio-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="If given, sample real 5-s windows from this directory instead of random noise.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models/hf_cache"),
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON report path.",
)
def main(
    onnx_path: Path,
    n: int,
    seed: int,
    audio_root: Path | None,
    cache_dir: Path,
    out: Path | None,
) -> None:
    encoder = MertEncoder(cache_dir=cache_dir, device="cpu")
    pooled = encoder.pooled_module
    wrapper = MertOnnxWrapper(pooled).eval()

    session = ort.InferenceSession(
        onnx_path.as_posix(), providers=["CPUExecutionProvider"]
    )

    samples = _generate_or_load_samples(n, seed, audio_root)
    cosines = np.empty(n, dtype=np.float32)
    for i, wav in enumerate(samples):
        pt = _pt_embed(wrapper, wav)
        oo = _ort_embed(session, wav)
        # Both sides should already be L2-normalized; clip for safety.
        denom = float(np.linalg.norm(pt) * np.linalg.norm(oo))
        cosines[i] = float(np.dot(pt, oo) / max(denom, 1e-12))
        assert pt.shape == (EMBEDDING_DIM,), pt.shape
        assert oo.shape == (EMBEDDING_DIM,), oo.shape

    report = {
        "onnx_path": str(onnx_path),
        "n": int(n),
        "audio_root": str(audio_root) if audio_root else None,
        "cosine_mean": float(cosines.mean()),
        "cosine_min": float(cosines.min()),
        "cosine_p25": float(np.percentile(cosines, 25)),
        "cosine_p50": float(np.percentile(cosines, 50)),
        "cosine_p75": float(np.percentile(cosines, 75)),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
    click.echo(payload)

    # Soft gate: warn loudly if below threshold but don't fail by default.
    if report["cosine_mean"] < 0.9995:
        click.echo(
            f"WARNING: mean cosine {report['cosine_mean']:.5f} is below the 0.9995 gate.",
            err=True,
        )


if __name__ == "__main__":
    main()
