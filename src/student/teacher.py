"""LAION-CLAP teacher wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from student.config import CLAP_SR, CLAP_WINDOW_SAMPLES
from student.data import load_window


class ClapTeacher:
    """Thin wrapper around ``laion_clap.CLAP_Module`` for audio embeddings."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        device: str = "cpu",
        amodel: str = "HTSAT-base",
        enable_fusion: bool = False,
    ):
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"CLAP checkpoint not found at {ckpt}. Run `make download-clap-music` "
                "or pass --teacher-checkpoint."
            )

        try:
            import laion_clap  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                "laion-clap is required for teacher embedding. "
                "Install with `python -m pip install -e .[train]`."
            ) from e

        self.checkpoint = ckpt
        self.device = device
        self.amodel = amodel
        self.enable_fusion = enable_fusion
        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion,
            amodel=amodel,
            device=device,
        )
        self.model.load_ckpt(str(ckpt), verbose=False)
        self.model.eval()

    def embed_waveforms(self, waveforms_48k: list[np.ndarray]) -> np.ndarray:
        if not waveforms_48k:
            return np.empty((0, 0), dtype=np.float32)
        batch = np.stack(waveforms_48k, axis=0).astype(np.float32, copy=False)
        embeddings = self.model.get_audio_embedding_from_data(batch, use_tensor=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        return embeddings / norms


def load_clap_window(path: str | Path, start_sample_24k: int) -> np.ndarray:
    start_48k = int(round(start_sample_24k * (CLAP_SR / 24_000.0)))
    return load_window(
        path,
        start_48k,
        target_sr=CLAP_SR,
        window_samples=CLAP_WINDOW_SAMPLES,
    )
