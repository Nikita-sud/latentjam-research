"""Export ``MertPooledEncoder`` + input normalization to ONNX.

Why bake normalization into the graph: HuggingFace's ``Wav2Vec2FeatureExtractor``
applies per-sample zero-mean unit-variance normalization before feeding the
model. The on-device runtime won't have that processor available, so we add
the same normalization as a graph op. Callers (Python, Android) feed raw
waveform in ``[-1, 1]`` and get back a 768-d L2-normalized embedding.
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from torch import nn

from encoder.audio_io import TARGET_SR
from encoder.mert_encoder import (
    MODEL_VERSION,
    WINDOW_SAMPLES,
    MertEncoder,
    MertPooledEncoder,
)


class MertOnnxWrapper(nn.Module):
    """Adds the missing input-normalization step in front of ``MertPooledEncoder``.

    The wrapper operates on a fixed-shape input: ``(B, WINDOW_SAMPLES)`` raw
    waveform float32 in ``[-1, 1]``. Output: ``(B, 768)`` L2-normalized.
    """

    def __init__(self, pooled: MertPooledEncoder, eps: float = 1e-7):
        super().__init__()
        self.pooled = pooled
        self.eps = eps

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Wav2Vec2 normalization: per-sample zero mean, unit variance.
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True)
        normed = (waveform - mean) / (std + self.eps)
        return self.pooled(normed)


def export(
    out_path: str | Path,
    *,
    opset: int = 17,
    cache_dir: str | Path | None = "models/hf_cache",
) -> Path:
    """Export the wrapped MERT encoder to ONNX. Returns the output path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    encoder = MertEncoder(cache_dir=cache_dir, device="cpu")
    pooled = encoder.pooled_module
    wrapper = MertOnnxWrapper(pooled).eval()

    dummy = torch.zeros(1, WINDOW_SAMPLES, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy,),
        out.as_posix(),
        input_names=["waveform"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # fully static for mobile-friendly inference
        export_params=True,
    )

    metadata_path = out.with_suffix(".meta.txt")
    metadata_path.write_text(
        "\n".join(
            [
                f"model_version={MODEL_VERSION}",
                f"sample_rate={TARGET_SR}",
                f"window_samples={WINDOW_SAMPLES}",
                "input_name=waveform",
                f"input_shape=(1, {WINDOW_SAMPLES})",
                "output_name=embedding",
                "output_shape=(1, 768)",
                "input_normalization=zero-mean unit-variance per sample",
                "output=L2-normalized",
            ]
        )
        + "\n"
    )
    return out


@click.command()
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("models/onnx/mert_v1_95m_5s.onnx"),
    show_default=True,
)
@click.option("--opset", type=int, default=17, show_default=True)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models/hf_cache"),
    show_default=True,
)
def main(out_path: Path, opset: int, cache_dir: Path) -> None:
    """Export MERT-pooled to ONNX (FP32, fixed shape)."""
    out = export(out_path, opset=opset, cache_dir=cache_dir)
    click.echo(f"Wrote {out}")
    click.echo(f"Wrote {out.with_suffix('.meta.txt')}")


if __name__ == "__main__":
    main()
