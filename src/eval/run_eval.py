"""Evaluation orchestrator: load store + dataset, compute metrics, emit JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

from eval.metrics import album_cohesion, recall_at_k_genre, tag_jaccard_at_k
from predictor.store import EmbeddingStore


def _embedding_matrix(rows: list) -> np.ndarray:
    return np.stack([np.asarray(e, dtype=np.float32) for e in rows], axis=0)


@click.command()
@click.option(
    "--store",
    "store_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--dataset",
    type=click.Choice(["fma_small", "mtat", "library"], case_sensitive=False),
    required=True,
    help='Which content-only proxy to run.',
)
@click.option(
    "--audio-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Dataset audio root (defaults: data/raw/fma_small or data/raw/magnatagatune/mp3).",
)
@click.option(
    "--metadata-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help='Dataset metadata root (FMA: data/raw/fma_metadata; MTAT: data/raw/magnatagatune).',
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help='Where to write the JSON report (default: stdout).',
)
def main(
    store_path: Path,
    dataset: str,
    audio_root: Path | None,
    metadata_root: Path | None,
    out: Path | None,
) -> None:
    store = EmbeddingStore.open(store_path)
    if len(store) == 0:
        raise click.ClickException(f"store at {store_path} is empty")

    df = store.df.copy()
    ds = dataset.lower()
    report: dict = {"store_path": str(store_path), "dataset": ds, "n_store": len(store)}

    if ds == "fma_small":
        from eval.fma import align_store_to_fma, load_fma_index

        a_root = audio_root or Path("data/raw/fma_small")
        m_root = metadata_root or Path("data/raw/fma_metadata")
        idx = load_fma_index(a_root, m_root, subset="small")
        joined = align_store_to_fma(df, idx)
        if joined.empty:
            raise click.ClickException(
                "no rows joined to FMA-small. Did you embed `data/raw/fma_small`?"
            )
        matrix = _embedding_matrix(joined["embedding"].to_list())
        labels = joined["genre_top"].astype(str).to_list()
        recall = recall_at_k_genre(matrix, labels, k_values=(1, 5, 10, 20))
        report.update(
            n_aligned=int(len(joined)),
            unique_genres=int(joined["genre_top"].nunique()),
            recall_at_k_genre={str(k): v for k, v in recall.items()},
        )

    elif ds == "mtat":
        from eval.mtat import align_store_to_mtat, load_mtat_index

        a_root = audio_root or Path("data/raw/magnatagatune/mp3")
        m_root = metadata_root or Path("data/raw/magnatagatune")
        idx = load_mtat_index(a_root, m_root)
        joined = align_store_to_mtat(df, idx)
        if joined.empty:
            raise click.ClickException(
                "no rows joined to MagnaTagATune. Did you embed the dataset's mp3/ root?"
            )
        matrix = _embedding_matrix(joined["embedding"].to_list())
        tag_matrix = idx.tag_matrix_for(joined["mp3_path"].to_list())
        jaccard = tag_jaccard_at_k(matrix, tag_matrix, k_values=(1, 5, 10, 20))
        report.update(
            n_aligned=int(len(joined)),
            n_tags=int(tag_matrix.shape[1]),
            tag_jaccard_at_k={str(k): v for k, v in jaccard.items()},
        )

    elif ds == "library":
        # User's own collection: album cohesion using store's `album` column.
        df = df.dropna(subset=["album"])
        if df.empty:
            raise click.ClickException(
                "no rows have album metadata; cannot run album-cohesion eval."
            )
        matrix = _embedding_matrix(df["embedding"].to_list())
        result = album_cohesion(matrix, df["album"].astype(str).to_list())
        report.update(album_cohesion=result)

    payload = json.dumps(report, indent=2, sort_keys=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n")
    else:
        sys.stdout.write(payload + "\n")


if __name__ == "__main__":
    main()
