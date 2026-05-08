"""Export the mel-CNN student to ONNX."""

from __future__ import annotations

from pathlib import Path

import click
import torch

from student.benchmark import _load_checkpoint
from student.config import DEFAULT_STUDENT_CKPT, STUDENT_MODEL_VERSION, MelConfig


def export(checkpoint: str | Path, out_path: str | Path, *, opset: int = 17) -> Path:
    ckpt = Path(checkpoint)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = _load_checkpoint(ckpt, torch.device("cpu")).eval()
    mel_cfg = MelConfig()
    dummy = torch.zeros(1, 1, mel_cfg.n_mels, 1 + mel_cfg.window_samples // mel_cfg.hop_length)
    torch.onnx.export(
        model,
        (dummy,),
        out.as_posix(),
        input_names=["log_mel"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        export_params=True,
        dynamo=False,
    )
    out.with_suffix(".meta.txt").write_text(
        "\n".join(
            [
                f"model_version={STUDENT_MODEL_VERSION}",
                f"checkpoint={ckpt}",
                "input_name=log_mel",
                f"input_shape=(1, 1, {mel_cfg.n_mels}, {dummy.shape[-1]})",
                "output_name=embedding",
                f"output_shape=(1, {model.embedding_dim})",
                "input=normalized log-mel spectrogram",
                "output=L2-normalized CLAP-space embedding",
            ]
        )
        + "\n"
    )
    return out


@click.command()
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
    default=Path("models/onnx/mel_cnn_student.onnx"),
    show_default=True,
)
@click.option("--opset", type=int, default=17, show_default=True)
def main(checkpoint_path: Path, out_path: Path, opset: int) -> None:
    out = export(checkpoint_path, out_path, opset=opset)
    click.echo(f"Wrote {out}")
    click.echo(f"Wrote {out.with_suffix('.meta.txt')}")


if __name__ == "__main__":
    main()
