"""CLI: ``python -m predictor.cli embed|recommend``."""

from __future__ import annotations

from pathlib import Path

import click
from tqdm import tqdm

from encoder.audio_io import AudioLoadError
from encoder.batching import iterate_files
from encoder.cache import content_fingerprint
from predictor.store import EmbeddingStore, TrackRecord
from predictor.tags import read_tags


@click.group()
def cli() -> None:
    """latentjam-research embedding store CLI."""


@cli.command()
@click.option(
    "--root",
    "root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Audio library root (recursively scanned).",
)
@click.option(
    "--out",
    "out",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output parquet path. Resumed if it already exists.",
)
@click.option("--device", default="cpu", show_default=True, help='"cpu" or "cuda".')
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after N new files (useful for smoke tests).",
)
@click.option(
    "--save-every",
    type=int,
    default=200,
    show_default=True,
    help="Persist the store every N successful embeddings.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("models/hf_cache"),
    show_default=True,
    help="HuggingFace cache directory for the MERT weights.",
)
def embed(
    root: Path,
    out: Path,
    device: str,
    limit: int | None,
    save_every: int,
    cache_dir: Path,
) -> None:
    """Walk ROOT, embed new audio files, write to OUT (parquet)."""
    # Heavy imports deferred so `--help` is fast.
    from encoder.mert_encoder import MODEL_VERSION, MertEncoder

    out.parent.mkdir(parents=True, exist_ok=True)
    store = EmbeddingStore.open(out)
    encoder = MertEncoder(device=device, cache_dir=cache_dir)

    files = list(iterate_files(root))
    click.echo(f"Found {len(files)} candidate audio files under {root}")

    new_count = 0
    skipped_existing = 0
    failed = 0

    pbar = tqdm(files, desc="embedding", unit="file")
    try:
        for path in pbar:
            try:
                tid = content_fingerprint(path)
            except OSError as e:
                click.echo(f"\n[skip] cannot read {path}: {e}", err=True)
                failed += 1
                continue

            if store.contains(tid):
                skipped_existing += 1
                continue

            try:
                emb = encoder.embed_file(path)
            except (AudioLoadError, RuntimeError) as e:
                click.echo(f"\n[skip] embed failed for {path}: {e}", err=True)
                failed += 1
                continue

            title, artist, album, genre, year = read_tags(path)
            store.add(
                TrackRecord(
                    track_id=tid,
                    path=str(path.resolve()),
                    embedding=emb,
                    model_version=MODEL_VERSION,
                    title=title,
                    artist=artist,
                    album=album,
                    genre=genre,
                    year=year,
                )
            )
            new_count += 1

            if save_every > 0 and new_count % save_every == 0:
                store.save(out)

            if limit is not None and new_count >= limit:
                break
    finally:
        store.save(out)
        pbar.close()

    click.echo(
        f"Done. new={new_count} skipped_existing={skipped_existing} failed={failed} "
        f"total_in_store={len(store)}"
    )


@cli.command()
@click.option(
    "--store",
    "store_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--seed-id", "seed_id", type=str, required=True, help="track_id of the seed.")
@click.option("-k", type=int, default=10, show_default=True)
@click.option(
    "--include-self",
    is_flag=True,
    default=False,
    help="Include the seed track in results (default: excluded).",
)
def recommend(store_path: Path, seed_id: str, k: int, include_self: bool) -> None:
    """Print the top-k cosine-nearest neighbors of SEED_ID."""
    store = EmbeddingStore.open(store_path)
    if not store.contains(seed_id):
        raise click.ClickException(f"seed-id {seed_id!r} not found in store {store_path}")

    hits = store.search_by_id(seed_id, k=k, exclude_self=not include_self)
    for h in hits:
        meta = " - ".join(
            x for x in (h.artist, h.title, h.album) if x
        ) or h.path
        click.echo(f"{h.rank:>3}  {h.score:+.4f}  {h.track_id}  {meta}")


if __name__ == "__main__":
    cli()
