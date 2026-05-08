"""INT8 dynamic quantization of the exported MERT ONNX model.

Dynamic quantization quantizes weights once (offline) and computes activation
scales on the fly per inference. No calibration data needed; quality loss is
typically <0.5 % on transformer encoders. If we ever need static int8 (smaller,
faster) we'll need a calibration set — that's a v0.5 task.
"""

from __future__ import annotations

from pathlib import Path

import click
from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize(in_path: str | Path, out_path: str | Path) -> Path:
    inp = Path(in_path)
    out = Path(out_path)
    if not inp.exists():
        raise FileNotFoundError(f"input ONNX not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=inp.as_posix(),
        model_output=out.as_posix(),
        weight_type=QuantType.QInt8,
    )

    meta_src = inp.with_suffix(".meta.txt")
    if meta_src.exists():
        out.with_suffix(".meta.txt").write_text(
            meta_src.read_text() + "quantization=dynamic-int8 (weights only)\n"
        )
    return out


@click.command()
@click.option(
    "--in",
    "in_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
def main(in_path: Path, out_path: Path) -> None:
    out = quantize(in_path, out_path)
    in_size = in_path.stat().st_size / 1e6
    out_size = out.stat().st_size / 1e6
    click.echo(f"Wrote {out} ({out_size:.1f} MB; was {in_size:.1f} MB)")


if __name__ == "__main__":
    main()
