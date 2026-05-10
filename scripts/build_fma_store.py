"""Build an EmbeddingStore parquet for FMA-medium using a trained SSL student.

This is the canonical FMA-medium store the predictor (session model) trains
on. Each track gets one 5-s window (middle of the track) embedded by the
student and stored with its FMA metadata (artist, album, track number,
genre, year). Album order is preserved via ``track_number`` so the
predictor can reconstruct album-as-session.

Run:
    python scripts/build_fma_store.py \\
        --audio-root data/raw/fma_medium \\
        --metadata-root data/raw/fma_metadata \\
        --subset medium \\
        --checkpoint models/student/mel_cnn_ssl_fma_medium_v3.pt \\
        --out models/store/fma_medium_v3.parquet \\
        --device mps
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from predictor.store import EmbeddingStore, TrackRecord
from student.benchmark import _load_checkpoint
from student.config import MelConfig, TARGET_SR, WINDOW_SAMPLES, WINDOW_SECONDS
from student.data import _ffmpeg_decode_window, load_fma_manifest
from student.device import resolve_torch_device
from student.mel import LogMelExtractor


def _safe_year(value) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # FMA's album.date_released is YYYY-MM-DD or empty.
    head = s[:4]
    if head.isdigit():
        return int(head)
    return None


def _decode_one(args):
    fma_track_id, path, start_sample, duration_samples = args
    wav = _ffmpeg_decode_window(
        path,
        start_sample,
        target_sr=TARGET_SR,
        window_samples=duration_samples,
    )
    if wav is None or wav.size == 0:
        return fma_track_id, None
    out = np.zeros(duration_samples, dtype=np.float32)
    n = min(wav.shape[0], duration_samples)
    out[:n] = wav[:n]
    return fma_track_id, out


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
    "--checkpoint",
    "checkpoint_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--device", default="auto", show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option(
    "--num-workers",
    type=int,
    default=8,
    show_default=True,
    help="ffmpeg decode threads. subprocess.run releases the GIL while waiting.",
)
@click.option("--limit", type=int, default=None, help="Cap number of tracks (debug).")
def main(
    audio_root: Path,
    metadata_root: Path,
    subset: str,
    checkpoint_path: Path,
    out_path: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    limit: int | None,
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
        raise click.ClickException("empty manifest; check --audio-root / --subset")
    n_tracks = len(manifest)
    print(f"manifest: {n_tracks} tracks; subset={subset}; device={torch_device}")

    # Mid-track 5s window per track (window_starts(...) returns max_start//2 for w=1).
    window_samples = int(WINDOW_SAMPLES)
    decode_args = []
    track_ids = []
    for row in manifest.itertuples(index=False):
        # Best-effort: read duration via the patched _ffmpeg_decode_window probe;
        # to keep the planning step cheap we just decode middle assuming >=5 s.
        # Tracks shorter than 5 s will be padded with zeros by _decode_one.
        # FMA-medium tracks are 30 s each, so this is safe.
        start_sample = 0  # decoded chunk is 5 s; we'll later seek inside ffmpeg
        # Use middle-ish offset: 12.5 s in. Still well within FMA-medium 30 s clips.
        start_sample = max(0, int(12.5 * TARGET_SR) - window_samples // 2)
        decode_args.append((int(row.fma_track_id), str(row.path), start_sample, window_samples))
        track_ids.append(int(row.fma_track_id))

    print("loading student checkpoint...")
    model = _load_checkpoint(checkpoint_path, torch_device).eval()
    mel = LogMelExtractor(MelConfig()).to(torch_device).eval()
    ckpt_meta = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_version = str(ckpt_meta.get("model_version", "unknown"))
    print(f"model_version={model_version}")

    # Index manifest rows by fma_track_id for metadata lookup at write time.
    by_id = manifest.set_index("fma_track_id")

    # Decode in parallel via threads (ffmpeg releases GIL during subprocess wait),
    # then batch into the model on the device.
    waveforms: dict[int, np.ndarray] = {}
    skipped = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for tid, wav in tqdm(
            pool.map(_decode_one, decode_args),
            total=len(decode_args),
            desc="decode",
            unit="track",
        ):
            if wav is None:
                skipped += 1
                continue
            waveforms[tid] = wav

    print(f"decoded {len(waveforms)}/{len(decode_args)} tracks ({skipped} skipped)")

    # Forward through student in batches.
    embeddings: dict[int, np.ndarray] = {}
    valid_ids = list(waveforms.keys())
    with torch.inference_mode():
        for i in tqdm(range(0, len(valid_ids), batch_size), desc="embed", unit="batch"):
            batch_ids = valid_ids[i : i + batch_size]
            batch = np.stack([waveforms[t] for t in batch_ids], axis=0)
            wav_t = torch.from_numpy(batch).to(torch_device)
            mel_t = mel(wav_t)
            emb = model(mel_t)
            emb_np = emb.cpu().numpy().astype(np.float32, copy=False)
            for tid, vec in zip(batch_ids, emb_np, strict=True):
                norm = max(float(np.linalg.norm(vec)), 1e-12)
                embeddings[tid] = vec / norm

    print(f"embedded {len(embeddings)} tracks")

    store = EmbeddingStore()
    for tid in valid_ids:
        meta = by_id.loc[tid] if tid in by_id.index else None
        if meta is None:
            continue
        artist = meta.get("artist_name") if isinstance(meta, pd.Series) else None
        album = meta.get("album_title") if isinstance(meta, pd.Series) else None
        title = meta.get("track_title") if isinstance(meta, pd.Series) else None
        genre = meta.get("genre_top") if isinstance(meta, pd.Series) else None
        year = _safe_year(meta.get("album_date_released") if isinstance(meta, pd.Series) else None)
        track_number = meta.get("track_number") if isinstance(meta, pd.Series) else None
        try:
            track_number_int = int(track_number) if not pd.isna(track_number) else None
        except (TypeError, ValueError):
            track_number_int = None
        try:
            album_id = int(meta["album_id"]) if not pd.isna(meta["album_id"]) else None
        except (TypeError, ValueError, KeyError):
            album_id = None
        try:
            artist_id = int(meta["artist_id"]) if not pd.isna(meta["artist_id"]) else None
        except (TypeError, ValueError, KeyError):
            artist_id = None
        path_str = str(meta["path"]) if "path" in meta else ""
        record = TrackRecord(
            track_id=f"fma-{int(tid):06d}",
            path=path_str,
            embedding=embeddings[tid],
            model_version=model_version,
            title=str(title) if title and not pd.isna(title) else None,
            artist=str(artist) if artist and not pd.isna(artist) else None,
            album=str(album) if album and not pd.isna(album) else None,
            genre=str(genre) if genre and not pd.isna(genre) else None,
            year=year,
            extra={
                "fma_track_id": int(tid),
                "album_id": album_id,
                "artist_id": artist_id,
                "track_number": track_number_int,
                "split": str(meta.get("split", "")) if isinstance(meta, pd.Series) else "",
            },
        )
        store.add(record)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(out_path)

    # Side-load the extras into a sidecar parquet so the predictor can read
    # album_id / track_number without changing the canonical store schema.
    extras_path = out_path.with_suffix(".extras.parquet")
    rows = []
    for tid in valid_ids:
        if tid not in embeddings:
            continue
        meta = by_id.loc[tid]
        try:
            album_id = int(meta["album_id"]) if not pd.isna(meta["album_id"]) else None
        except (TypeError, ValueError, KeyError):
            album_id = None
        try:
            artist_id = int(meta["artist_id"]) if not pd.isna(meta["artist_id"]) else None
        except (TypeError, ValueError, KeyError):
            artist_id = None
        try:
            tnum = int(meta["track_number"]) if not pd.isna(meta["track_number"]) else None
        except (TypeError, ValueError, KeyError):
            tnum = None
        rows.append(
            {
                "track_id": f"fma-{int(tid):06d}",
                "fma_track_id": int(tid),
                "album_id": album_id,
                "artist_id": artist_id,
                "track_number": tnum,
                "split": str(meta.get("split", "")),
            }
        )
    pd.DataFrame(rows).to_parquet(extras_path, index=False)

    report = {
        "store": str(out_path),
        "extras": str(extras_path),
        "n_tracks": int(len(store)),
        "n_skipped": int(skipped),
        "model_version": model_version,
        "embedding_dim": int(store.dim or 0),
        "device": str(torch_device),
        "subset": subset,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
