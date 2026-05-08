"""MERT-v1-95M wrapper producing a single L2-normalized 768-d embedding per track.

The recipe is pinned by ``CLAUDE.md`` and must stay in sync with the ONNX export
wrapper in ``src/conversion/export_onnx.py``: any change to the pooling layers,
window size, or normalization bumps the ``MODEL_VERSION`` suffix.

Note: ``m-a-p/MERT-v1-95M`` requires ``trust_remote_code=True`` because the
model class is custom (HuBERT variant). License is **CC-BY-NC-4.0** — research
use only; do not redistribute weights inside a shipping app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn

from encoder.audio_io import TARGET_SR, load_audio, windows

MODEL_ID: Final[str] = "m-a-p/MERT-v1-95M"
MODEL_VERSION: Final[str] = "mert-v1-95m@layers5-8/meanpool/v1"
EMBEDDING_DIM: Final[int] = 768
WINDOW_SECONDS: Final[float] = 5.0
WINDOW_SAMPLES: Final[int] = int(WINDOW_SECONDS * TARGET_SR)  # 120_000
HOP_SAMPLES: Final[int] = WINDOW_SAMPLES // 2  # 50% overlap

# Pooling indices into the 13-element ``hidden_states`` tuple returned by MERT
# (index 0 = feature-extractor output, 1..12 = transformer layers). Indices
# [5, 6, 7, 8] are the music-similarity sweet spot per MARBLE.
POOL_LAYER_INDICES: Final[tuple[int, ...]] = (5, 6, 7, 8)


class MertPooledEncoder(nn.Module):
    """Wraps MERT with the pinned mean-pool / L2-norm head as a single nn.Module.

    Exposed as a Module (not just a function) so that ``torch.onnx.export`` can
    capture the full pooling graph rather than relying on Python-side pooling.
    """

    def __init__(self, model: nn.Module, layer_indices: tuple[int, ...] = POOL_LAYER_INDICES):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_values=input_values, output_hidden_states=True)
        # hidden_states is a tuple of (B, T, D). Stack only the layers we want
        # to keep the graph small.
        selected = torch.stack([outputs.hidden_states[i] for i in self.layer_indices], dim=0)
        # selected: (L, B, T, D) -> mean over L -> (B, T, D)
        x = selected.mean(dim=0)
        # mean over time -> (B, D)
        x = x.mean(dim=1)
        # L2-normalize -> (B, D)
        x = nn.functional.normalize(x, p=2.0, dim=-1)
        return x


class MertEncoder:
    """High-level helper that loads MERT lazily and embeds files or waveforms."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self._pooled: MertPooledEncoder | None = None
        self._processor = None  # type: ignore[assignment]

    def _ensure_loaded(self) -> None:
        if self._pooled is not None:
            return

        # Lazy import: keeps `from encoder.mert_encoder import MODEL_VERSION` cheap.
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        kwargs = {"trust_remote_code": True}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        model = AutoModel.from_pretrained(self.model_id, **kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        pooled = MertPooledEncoder(model).to(self.device)
        pooled.eval()
        self._pooled = pooled
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id, **kwargs)

    @property
    def pooled_module(self) -> MertPooledEncoder:
        """The pooled nn.Module — exposed for ONNX export."""
        self._ensure_loaded()
        assert self._pooled is not None
        return self._pooled

    def _preprocess(self, window: np.ndarray) -> torch.Tensor:
        """Run MERT's feature extractor on a single 5-s window."""
        assert self._processor is not None
        inputs = self._processor(
            window, sampling_rate=TARGET_SR, return_tensors="pt"
        )
        return inputs["input_values"].to(self.device)

    @torch.inference_mode()
    def embed_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Embed a 1-D waveform (24 kHz mono float32) into a 768-d vector.

        Tracks longer than 5 s are split into overlapping windows; per-window
        embeddings are mean-pooled and re-normalized.
        """
        if waveform.ndim != 1:
            raise ValueError(f"expected 1-D waveform, got shape {waveform.shape}")
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32, copy=False)

        self._ensure_loaded()
        chunks = windows(waveform, WINDOW_SAMPLES, HOP_SAMPLES)

        embeds = []
        for chunk in chunks:
            x = self._preprocess(chunk)
            assert self._pooled is not None
            emb = self._pooled(x)  # (1, 768), already L2-normed
            embeds.append(emb.squeeze(0).cpu().numpy())

        stacked = np.stack(embeds, axis=0)  # (n_windows, 768)
        track_emb = stacked.mean(axis=0)
        norm = float(np.linalg.norm(track_emb))
        if norm > 0:
            track_emb = track_emb / norm
        return track_emb.astype(np.float32, copy=False)

    def embed_file(self, path: str | Path) -> np.ndarray:
        """Convenience: load + embed in one call."""
        wav = load_audio(path)
        return self.embed_waveform(wav)
