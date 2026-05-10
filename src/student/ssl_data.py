"""Pre-decode FMA-small clips into RAM and serve two random crops per item.

Used by ``train_ssl.py``. Decode goes through ``ffmpeg`` subprocess instead of
soundfile/libmpg123 because FMA-small ships a non-trivial number of mp3s with
broken layer-3 dequantization tables that crash libmpg123 with SIGBUS — an
unrecoverable signal that try/except cannot catch. ffmpeg in a child process
isolates that failure: a SIGBUS kills the child, the main process keeps going.

Each subprocess emits raw mono float32 PCM at the target sample rate on
stdout; we read it directly into a numpy array. Decode parallelizes across
cores via a ProcessPoolExecutor (one persistent ffmpeg call per worker).
Bad files become ``None`` and are filtered out at dataset construction.

Memory: FMA-small full set is ~8000 clips × 30 s × 24 kHz × 4 B ≈ 23 GB.
Comfortable on a 64 GB Mac.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm

from student.config import TARGET_SR, WINDOW_SAMPLES


def _ffmpeg_decode(args: tuple[str, int, float]) -> Optional[np.ndarray]:
    """Decode one audio file via ffmpeg subprocess. Returns float32 mono @ ``target_sr``.

    Designed to be a top-level function so ``ProcessPoolExecutor`` can pickle it.
    Returns ``None`` for any failure (bad file, ffmpeg not found, timeout).
    """
    path, target_sr, timeout_s = args
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel", "error",
                "-i", path,
                "-f", "f32le",
                "-ac", "1",
                "-ar", str(target_sr),
                "-",
            ],
            capture_output=True,
            check=True,
            timeout=timeout_s,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    if not result.stdout:
        return None
    # frombuffer returns a non-writable view of the immutable bytes; copy()
    # gives torch.from_numpy a writable array and decouples it from stdout.
    arr = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if arr.size < 1024:
        return None
    return arr


def decode_all_clips(
    manifest: pd.DataFrame,
    *,
    target_sr: int = TARGET_SR,
    num_workers: int = 12,
    timeout_s: float = 30.0,
    desc: str = "decode",
) -> list[Optional[np.ndarray]]:
    """Return a list aligned with ``manifest`` rows; entries are float32 1-D arrays
    or ``None`` if decode failed."""
    paths = manifest["path"].tolist()
    args = [(p, target_sr, timeout_s) for p in paths]

    out: list[Optional[np.ndarray]] = [None] * len(paths)
    # Threads launch ffmpeg subprocesses in parallel. The ffmpeg child is the
    # process that gets SIGBUS / segfault on broken mp3s; the main Python
    # process is never at risk. ``subprocess.run`` releases the GIL while
    # waiting on the child, so threads scale well across cores.
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for idx, wav in enumerate(
            tqdm(pool.map(_ffmpeg_decode, args), total=len(paths), desc=desc, unit="clip")
        ):
            out[idx] = wav
    return out


def _manifest_cache_key(manifest: pd.DataFrame, target_sr: int) -> str:
    """Stable digest over the manifest's path list + sample rate."""
    h = hashlib.sha256()
    h.update(str(target_sr).encode())
    for path in manifest["path"].astype(str).tolist():
        h.update(path.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _save_clips_npz(clips: list[Optional[np.ndarray]], path: Path) -> None:
    """Atomically save a list of variable-length float32 clips to one .npz file.

    Layout: ``concat`` (1-D float32 of all valid clips concatenated), ``valid``
    (bool per slot), ``lengths`` (int64 per slot, 0 for None). Written to a
    sibling ``.tmp`` file then renamed into place — partial writes never leave
    a usable-looking but corrupted cache.
    """
    valid_mask = np.fromiter((c is not None for c in clips), dtype=bool, count=len(clips))
    lengths = np.fromiter(
        (int(c.shape[0]) if c is not None else 0 for c in clips),
        dtype=np.int64,
        count=len(clips),
    )
    valid_clips = [c for c in clips if c is not None]
    if valid_clips:
        concat = np.concatenate(valid_clips, axis=0).astype(np.float32, copy=False)
    else:
        concat = np.zeros(0, dtype=np.float32)

    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    np.savez(tmp_path, concat=concat, valid=valid_mask, lengths=lengths)
    # np.savez appends ``.npz`` if the path doesn't already end in it.
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    written.replace(path)


def _load_clips_npz(path: Path, expected_len: int) -> Optional[list[Optional[np.ndarray]]]:
    """Load the .npz cache. Returns ``None`` on any mismatch / IO failure."""
    try:
        with np.load(path) as data:
            valid = np.asarray(data["valid"]).astype(bool)
            lengths = np.asarray(data["lengths"]).astype(np.int64)
            concat = np.asarray(data["concat"]).astype(np.float32, copy=False)
    except Exception as exc:
        print(f"[cache] failed to load {path}: {exc!r}", flush=True)
        return None

    if valid.shape[0] != expected_len:
        print(
            f"[cache] {path} length mismatch ({valid.shape[0]} vs {expected_len})",
            flush=True,
        )
        return None
    total = int(lengths.sum())
    if concat.shape[0] != total:
        print(
            f"[cache] {path} concat length mismatch ({concat.shape[0]} vs {total})",
            flush=True,
        )
        return None

    out: list[Optional[np.ndarray]] = []
    cursor = 0
    for is_valid, ln in zip(valid.tolist(), lengths.tolist(), strict=True):
        if not is_valid:
            out.append(None)
            continue
        chunk = concat[cursor : cursor + ln].copy()
        cursor += ln
        out.append(chunk)
    return out


def cache_or_decode_clips(
    manifest: pd.DataFrame,
    *,
    cache_dir: Path | str | None,
    cache_label: str,
    target_sr: int = TARGET_SR,
    num_workers: int = 12,
    timeout_s: float = 30.0,
    desc: str = "decode",
) -> list[Optional[np.ndarray]]:
    """Decode + cache clips on disk so subsequent runs skip the ffmpeg pass.

    Cache key is a SHA-256 over (target_sr, manifest paths) so adding,
    removing, or reordering rows invalidates the cache. ``cache_label``
    distinguishes splits (``train`` vs ``val``) inside the same cache dir.
    Persistent format is .npz with atomic-rename writes, so a SIGKILL
    midway through ``np.savez`` never leaves a half-written cache visible.
    """
    if cache_dir is None:
        return decode_all_clips(
            manifest,
            target_sr=target_sr,
            num_workers=num_workers,
            timeout_s=timeout_s,
            desc=desc,
        )

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    digest = _manifest_cache_key(manifest, target_sr)
    cache_file = cache_path / f"{cache_label}_{target_sr}_{digest}.npz"

    if cache_file.exists():
        print(f"[cache] loading {cache_file}", flush=True)
        loaded = _load_clips_npz(cache_file, expected_len=len(manifest))
        if loaded is not None:
            return loaded
        print(f"[cache] {cache_file} unusable, re-decoding", flush=True)

    clips = decode_all_clips(
        manifest,
        target_sr=target_sr,
        num_workers=num_workers,
        timeout_s=timeout_s,
        desc=desc,
    )
    print(f"[cache] saving {cache_file}", flush=True)
    _save_clips_npz(clips, cache_file)
    print(f"[cache] saved {cache_file} ({cache_file.stat().st_size / 1e9:.1f} GB)", flush=True)
    return clips


class FmaSslDataset(Dataset):
    """Yields ``(view1, view2, genre_label)`` for a manifest aligned with ``clips``.

    Two pairing modes for view2, mixed by ``artist_positive_prob``:

    1. **Same-track** (default, prob = 1 - artist_positive_prob): two independent
       random crops of the SAME clip — invariance to time position only.
    2. **Artist-positive** (prob = artist_positive_prob): view2 is a random crop
       of a DIFFERENT clip by the same ``artist_id``. Forces the encoder to be
       invariant to track identity within an artist's catalogue, which surfaces
       artist-level audio content (timbre, production style, vocal identity)
       beyond what same-track augmentation alone reveals. Falls back to
       same-track when the seed has no artist-mate available.

    Use ``artist_positive_prob=0.0`` for deterministic eval (view-pair cosine
    measures pure aug invariance, no inter-track noise).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        clips: list[Optional[np.ndarray]],
        *,
        view_samples: int = WINDOW_SAMPLES,
        seed: int | None = None,
        artist_positive_prob: float = 0.0,
    ):
        if len(manifest) != len(clips):
            raise ValueError(
                f"manifest length {len(manifest)} != clips length {len(clips)}"
            )
        self.manifest = manifest.reset_index(drop=True)
        self.clips = clips
        self.view_samples = int(view_samples)
        self.artist_positive_prob = float(artist_positive_prob)
        self._rng = np.random.default_rng(seed)
        self.valid_idxs: list[int] = [
            i
            for i, c in enumerate(clips)
            if c is not None and isinstance(c, np.ndarray) and c.shape[0] >= self.view_samples // 2
        ]

        self._artist_to_idxs: dict[str, list[int]] = {}
        if self.artist_positive_prob > 0 and "artist_id" in self.manifest.columns:
            for idx in self.valid_idxs:
                aid = self.manifest.iloc[idx].get("artist_id")
                if aid is None:
                    continue
                if isinstance(aid, float) and np.isnan(aid):
                    continue
                key = str(aid)
                self._artist_to_idxs.setdefault(key, []).append(idx)

    @property
    def artist_pair_pool_size(self) -> int:
        """Count of valid indexes whose artist has at least one other track."""
        return sum(
            len(idxs) for idxs in self._artist_to_idxs.values() if len(idxs) > 1
        )

    def reseed(self, seed: int) -> None:
        """Reset the per-process RNG used for crop and artist-positive sampling."""
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.valid_idxs)

    def _crop(self, wav: np.ndarray) -> np.ndarray:
        n = wav.shape[0]
        if n <= self.view_samples:
            out = np.zeros(self.view_samples, dtype=np.float32)
            out[:n] = wav
            return out
        start = int(self._rng.integers(0, n - self.view_samples + 1))
        return wav[start : start + self.view_samples].astype(np.float32, copy=False)

    def _maybe_artist_partner(self, real_idx: int) -> Optional[int]:
        if self.artist_positive_prob <= 0:
            return None
        if self._rng.random() >= self.artist_positive_prob:
            return None
        aid = self.manifest.iloc[real_idx].get("artist_id")
        if aid is None or (isinstance(aid, float) and np.isnan(aid)):
            return None
        candidates = self._artist_to_idxs.get(str(aid), [])
        if len(candidates) <= 1:
            return None
        # Sample uniformly from same-artist tracks excluding self.
        choice = int(self._rng.integers(0, len(candidates) - 1))
        partner = candidates[choice] if candidates[choice] != real_idx else candidates[-1]
        return partner

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        real_idx = self.valid_idxs[idx]
        wav1 = self.clips[real_idx]
        assert wav1 is not None

        partner_idx = self._maybe_artist_partner(real_idx)
        if partner_idx is not None:
            wav2 = self.clips[partner_idx]
            assert wav2 is not None
        else:
            wav2 = wav1

        view1 = self._crop(wav1)
        view2 = self._crop(wav2)
        label = str(self.manifest.iloc[real_idx].get("genre_top", "unknown"))
        return (
            torch.from_numpy(np.ascontiguousarray(view1)),
            torch.from_numpy(np.ascontiguousarray(view2)),
            label,
        )


def collate_ssl(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    v1, v2, labels = zip(*batch, strict=True)
    return torch.stack(list(v1), dim=0), torch.stack(list(v2), dim=0), list(labels)


def seed_ssl_worker(_worker_id: int) -> None:
    """Give each DataLoader worker's dataset copy a distinct numpy RNG seed."""
    worker = get_worker_info()
    if worker is None:
        return
    dataset = worker.dataset
    if hasattr(dataset, "reseed"):
        dataset.reseed(torch.initial_seed() % (2**32))
