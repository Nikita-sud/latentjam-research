"""Build data/synth/manifest.parquet from the synced on-device caches.

Usage:

    python scripts/synth_build_manifest.py \\
        --music-cache /path/to/music_cache.db \\
        --playback-db /path/to/playback_persistence-*.db
"""

from __future__ import annotations

from pathlib import Path

import click

from synth.manifest import build_manifest


@click.command()
@click.option(
    "--music-cache",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="music_cache.db synced from the phone (contains CachedFileData).",
)
@click.option(
    "--playback-db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="playback_persistence-*.db (contains TrackEmbeddingEntity).",
)
@click.option(
    "--audit-csv",
    default="data/manifests/music_audit_full_tags.csv",
    type=click.Path(dir_okay=False),
    help="Optional audited-tags CSV used to backfill null genre/language.",
)
@click.option(
    "--out",
    default="data/synth/manifest.parquet",
    type=click.Path(dir_okay=False),
    help="Output parquet path.",
)
def main(music_cache: str, playback_db: str, audit_csv: str, out: str) -> None:
    audit = audit_csv if Path(audit_csv).exists() else None
    df = build_manifest(music_cache, playback_db, audit)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    n = len(df)
    energy_frac = df["energy"].notna().mean() if n else 0.0
    click.echo(f"wrote {n} tracks -> {out_path}")
    click.echo(f"  non-null energy: {df['energy'].notna().sum()}/{n} ({energy_frac:.2%})")
    click.echo(f"  missing genre:   {df['genre'].isna().sum()}/{n}")
    click.echo(f"  missing artist:  {df['artist'].isna().sum()}/{n}")


if __name__ == "__main__":
    main()
