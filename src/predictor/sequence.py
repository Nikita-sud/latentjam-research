"""Session-context predictor: given the user's recent plays, predict the embedding
of the next track they'd want.

Inference shape (what the app will eventually call):

    pred = model(context_embeddings)         # (B, K, D) -> (B, D)
    scores = store_matrix @ pred.T           # cosine vs. all tracks
    topk = top-N scores, exclude already-played

Training data is currently *synthesized* from FMA album metadata, not real
listening history (we don't have any yet — the Android app's
``ListeningEventRecorder`` is v0.5+). For each album with >= K+1 tracks we
pick K context tracks at random and treat one held-out track as target.
This is a proxy for "tracks people listen together"; the model learns the
album-direction in latent space. Real history will replace this when it
arrives — the rest of the pipeline stays unchanged.

Model: residual MLP over stacked context embeddings.
- input:  (B, K, D)   K=4, D=512 by default
- output: (B, D)      L2-normalized
- params: ~3 M, runs trivially on CPU

Loss: in-batch InfoNCE on cosine similarities (positives = target track of
the same row; negatives = target tracks of all other rows in the batch),
plus a small direct cosine loss to anchor the prediction near the target.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from predictor.store import EmbeddingStore

PREDICTOR_VERSION = "session-mlp@k4-d512/v1"


@dataclass
class PredictorConfig:
    context_k: int = 4
    embedding_dim: int = 512
    hidden_dim: int = 1024
    dropout: float = 0.1
    residual_scale: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_k": self.context_k,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "residual_scale": self.residual_scale,
            "predictor_version": PREDICTOR_VERSION,
        }


class SessionContextPredictor(nn.Module):
    """Stack K context embeddings -> residual MLP -> next-track embedding.

    Mean-pool the context as the residual base; the MLP learns the offset
    (album/mood drift) on top. This is an inductive bias that holds for
    *related* tracks (album, artist, genre) and degrades gracefully when
    the context is heterogeneous.
    """

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.context_k * cfg.embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(cfg.residual_scale))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, K, D)
        b, k, d = context.shape
        assert k == self.cfg.context_k and d == self.cfg.embedding_dim
        flat = context.reshape(b, k * d)
        offset = self.mlp(flat)
        base = context.mean(dim=1)
        out = base + self.residual_scale * offset
        return F.normalize(out, p=2.0, dim=-1)


# ---------------------------------------------------------------------------
# Session synthesis from FMA album metadata
# ---------------------------------------------------------------------------


def _build_album_groups(
    store: EmbeddingStore, extras: pd.DataFrame, *, min_tracks: int
) -> dict[int, list[int]]:
    """Return {album_id: [row_indices into store, sorted by track_number]}.

    We attach store row indices (not track_ids) because the trainer slices
    the embedding matrix directly.
    """
    df = store.df.copy()
    df = df.reset_index(drop=False).rename(columns={"index": "row"})
    merged = df.merge(extras[["track_id", "album_id", "track_number"]], on="track_id", how="inner")
    merged = merged.dropna(subset=["album_id"])
    merged["album_id"] = merged["album_id"].astype(int)
    # track_number may be missing for some FMA rows; sort fallback by row.
    merged["track_number"] = merged["track_number"].fillna(-1).astype(int)
    out: dict[int, list[int]] = {}
    for album_id, group in merged.groupby("album_id"):
        ordered = group.sort_values(["track_number", "row"])
        rows = ordered["row"].tolist()
        if len(rows) >= min_tracks:
            out[int(album_id)] = rows
    return out


def _split_albums(
    albums: dict[int, list[int]], *, val_frac: float, seed: int
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    rng = np.random.default_rng(seed)
    keys = list(albums.keys())
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_frac)))
    val_keys = set(keys[:n_val])
    train = {k: v for k, v in albums.items() if k not in val_keys}
    val = {k: v for k, v in albums.items() if k in val_keys}
    return train, val


class AlbumSessionDataset(Dataset):
    """For each album: sample K context tracks + 1 target track per __getitem__.

    We do not emit a fixed list of (ctx, target) pairs and instead resample
    at every access — with only ~5k albums in FMA-medium this gives the
    optimizer way more variety than fixed enumeration. Order of context
    tracks does not carry meaning here (album cohabitation is a bag, not a
    sequence), so we sample the K context tracks without replacement and
    leave the model order-agnostic via mean-pool residual.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        albums: dict[int, list[int]],
        *,
        context_k: int,
        seed: int = 0,
    ):
        self.matrix = matrix
        self.album_ids = list(albums.keys())
        self.albums = albums
        self.context_k = context_k
        # Per-worker RNG seeded with object id so workers diverge.
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.album_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        album_id = self.album_ids[idx]
        rows = self.albums[album_id]
        n = len(rows)
        if n < self.context_k + 1:
            # Should not happen because we filtered by min_tracks at build time.
            raise IndexError(f"album {album_id} too small: {n}")
        chosen = self._rng.choice(n, size=self.context_k + 1, replace=False)
        target_row = rows[int(chosen[-1])]
        ctx_rows = [rows[int(i)] for i in chosen[:-1]]
        ctx = self.matrix[ctx_rows].astype(np.float32, copy=False)
        target = self.matrix[target_row].astype(np.float32, copy=False)
        return torch.from_numpy(ctx), torch.from_numpy(target)


def collate_session(batch):
    ctx, target = zip(*batch, strict=True)
    return torch.stack(list(ctx), dim=0), torch.stack(list(target), dim=0)


def _build_artist_groups(
    store: EmbeddingStore, extras: pd.DataFrame, *, min_tracks: int
) -> dict[int, list[int]]:
    """Group store rows by ``artist_id`` (sorted by track_number then row).

    Same-artist tracks across different albums approximate "you listen to
    this artist together"; weaker signal than album cohabitation but
    useful when albums are sparse.
    """
    df = store.df.reset_index(drop=False).rename(columns={"index": "row"})
    merged = df.merge(extras[["track_id", "artist_id", "track_number"]], on="track_id", how="inner")
    merged = merged.dropna(subset=["artist_id"])
    merged["artist_id"] = merged["artist_id"].astype(int)
    merged["track_number"] = merged["track_number"].fillna(-1).astype(int)
    out: dict[int, list[int]] = {}
    for artist_id, group in merged.groupby("artist_id"):
        ordered = group.sort_values(["track_number", "row"])
        rows = ordered["row"].tolist()
        if len(rows) >= min_tracks:
            out[int(artist_id)] = rows
    return out


class MixedSessionDataset(Dataset):
    """Sample (context, target) from a *mixture* of session sources.

    sources is a list of (groups_dict, weight) pairs; on each ``__getitem__``
    we draw a source by weight, then a group from that source uniformly.

    Intent: train the predictor on multi-signal "what plays together" rather
    than just album cohabitation. Album signal is strongest, artist signal
    is weaker but covers tracks whose album has < K+1 tracks.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        sources: list[tuple[dict[int, list[int]], float]],
        *,
        context_k: int,
        seed: int = 0,
        virtual_size: int | None = None,
    ):
        self.matrix = matrix
        self.context_k = context_k
        # Per-source flat (group_id, list-rows) plus prefix probabilities.
        self.sources: list[tuple[list[tuple[int, list[int]]], float]] = []
        total_w = 0.0
        for groups, w in sources:
            entries = [(gid, rows) for gid, rows in groups.items() if len(rows) >= context_k + 1]
            if entries and w > 0:
                self.sources.append((entries, float(w)))
                total_w += float(w)
        if not self.sources or total_w <= 0:
            raise ValueError("MixedSessionDataset needs at least one non-empty source")
        # Normalize source weights, then compute total positive groups (for length).
        norm_w = [w / total_w for _, w in self.sources]
        self._weights = np.asarray(norm_w, dtype=np.float64)
        n_pairs = sum(len(entries) for entries, _ in self.sources)
        self._virtual_size = int(virtual_size) if virtual_size is not None else n_pairs
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._virtual_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # idx is ignored under random sampling; PyTorch DataLoader still
        # passes one. We resample at every access for diversity.
        src_idx = int(self._rng.choice(len(self.sources), p=self._weights))
        entries, _ = self.sources[src_idx]
        gid, rows = entries[int(self._rng.integers(0, len(entries)))]
        n = len(rows)
        chosen = self._rng.choice(n, size=self.context_k + 1, replace=False)
        target_row = rows[int(chosen[-1])]
        ctx_rows = [rows[int(i)] for i in chosen[:-1]]
        ctx = self.matrix[ctx_rows].astype(np.float32, copy=False)
        target = self.matrix[target_row].astype(np.float32, copy=False)
        return torch.from_numpy(ctx), torch.from_numpy(target)


# ---------------------------------------------------------------------------
# Training + eval
# ---------------------------------------------------------------------------


def info_nce_loss(
    pred: torch.Tensor, target: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE on cosine similarities.

    Both inputs are L2-normalized along dim=-1 (we normalize defensively in
    case the caller forgot). Positive pairs are diagonal entries of the
    pred @ target.T matrix.
    """
    pred = F.normalize(pred, p=2.0, dim=-1)
    target = F.normalize(target, p=2.0, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    loss_p = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_p + loss_t)


@torch.inference_mode()
def evaluate_baseline(
    matrix: np.ndarray,
    albums: dict[int, list[int]],
    *,
    device: torch.device,
    context_k: int,
    n_samples_per_album: int = 4,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    mode: str = "centroid",
) -> dict[str, float]:
    """Naive retrieval baseline (no learned predictor).

    mode='centroid' — use L2-normalized mean of context as the query.
    mode='last'     — use the last context track as the query.
    """
    rng = np.random.default_rng(rng_seed)
    pool_t = torch.from_numpy(matrix).to(device)
    ranks: list[int] = []
    for album_id, rows in albums.items():
        n = len(rows)
        if n < context_k + 1:
            continue
        for _ in range(n_samples_per_album):
            chosen = rng.choice(n, size=context_k + 1, replace=False)
            target_row = rows[int(chosen[-1])]
            ctx_rows = [rows[int(i)] for i in chosen[:-1]]
            ctx = torch.from_numpy(matrix[ctx_rows].astype(np.float32)).to(device)
            if mode == "last":
                query = ctx[-1]
            else:
                query = ctx.mean(dim=0)
            query = F.normalize(query, p=2.0, dim=-1)
            scores = pool_t @ query
            scores[ctx_rows] = -1.0
            order = torch.argsort(scores, descending=True)
            target_pos = int((order == target_row).nonzero(as_tuple=False).item())
            ranks.append(target_pos + 1)
    ranks_arr = np.asarray(ranks, dtype=np.int64)
    out: dict[str, float] = {
        "n_samples": int(len(ranks_arr)),
        "mrr": float(np.mean(1.0 / np.maximum(ranks_arr, 1))),
        "median_rank": float(np.median(ranks_arr)),
        "mode": mode,
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks_arr <= k))
    return out


@torch.inference_mode()
def evaluate(
    model: SessionContextPredictor,
    matrix: np.ndarray,
    albums: dict[int, list[int]],
    *,
    device: torch.device,
    context_k: int,
    n_samples_per_album: int = 4,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    candidate_pool: np.ndarray | None = None,
) -> dict[str, float]:
    """Recall@k and MRR over held-out albums.

    For each album we sample ``n_samples_per_album`` (context, target) pairs
    and ask: where does the true target appear in the cosine-ranked list of
    ALL store embeddings (excluding the K context rows)?
    """
    rng = np.random.default_rng(rng_seed)
    pool = candidate_pool if candidate_pool is not None else matrix
    pool_t = torch.from_numpy(pool).to(device)  # (N, D)
    model.eval()

    ranks: list[int] = []
    cosines: list[float] = []
    for album_id, rows in albums.items():
        n = len(rows)
        if n < context_k + 1:
            continue
        for _ in range(n_samples_per_album):
            chosen = rng.choice(n, size=context_k + 1, replace=False)
            target_row = rows[int(chosen[-1])]
            ctx_rows = [rows[int(i)] for i in chosen[:-1]]
            ctx = torch.from_numpy(matrix[ctx_rows][None, :, :].astype(np.float32)).to(device)
            pred = model(ctx)  # (1, D)
            pred = F.normalize(pred, p=2.0, dim=-1)
            scores = (pool_t @ pred.T).squeeze(-1)  # (N,)
            scores[ctx_rows] = -1.0  # mask context rows so they don't crowd top
            order = torch.argsort(scores, descending=True)
            target_pos = int((order == target_row).nonzero(as_tuple=False).item())
            ranks.append(target_pos + 1)  # rank from 1
            cosines.append(float(scores[target_row].item()))

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    cos_arr = np.asarray(cosines, dtype=np.float64)
    out: dict[str, float] = {
        "n_samples": int(len(ranks_arr)),
        "mrr": float(np.mean(1.0 / np.maximum(ranks_arr, 1))),
        "median_rank": float(np.median(ranks_arr)),
        "mean_cosine_to_target": float(np.mean(cos_arr)),
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks_arr <= k))
    return out


@click.command()
@click.option(
    "--store",
    "store_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="EmbeddingStore parquet (from build_fma_store).",
)
@click.option(
    "--extras",
    "extras_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Sidecar parquet with album_id / track_number columns.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--context-k", type=int, default=4, show_default=True)
@click.option("--epochs", type=int, default=30, show_default=True)
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--temperature", type=float, default=0.07, show_default=True)
@click.option("--cosine-weight", type=float, default=0.25, show_default=True)
@click.option("--val-frac", type=float, default=0.1, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--samples-per-album-eval", type=int, default=4, show_default=True)
@click.option(
    "--multi-signal",
    is_flag=True,
    default=False,
    help="Mix album sessions with artist-discography sessions during training. "
    "Albums stay strong positives; same-artist cross-album tracks add coverage "
    "for tracks whose album has < K+1 entries. Eval still uses album sessions only.",
)
@click.option(
    "--artist-weight",
    type=float,
    default=0.3,
    show_default=True,
    help="Weight of artist-source vs album-source under --multi-signal "
    "(album weight is fixed at 1.0).",
)
def main(
    store_path: Path,
    extras_path: Path,
    out_path: Path,
    context_k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    cosine_weight: float,
    val_frac: float,
    device: str,
    seed: int,
    samples_per_album_eval: int,
    multi_signal: bool,
    artist_weight: float,
) -> None:
    from student.device import resolve_torch_device

    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    store = EmbeddingStore.open(store_path)
    if len(store) == 0:
        raise click.ClickException(f"empty store at {store_path}")
    extras = pd.read_parquet(extras_path)

    matrix = store._matrix  # (N, D), L2-normalized at write time
    if matrix is None:
        store.rebuild_matrix()
        matrix = store._matrix
    assert matrix is not None
    embedding_dim = int(matrix.shape[1])
    print(f"store: {len(store)} tracks, dim={embedding_dim}")

    albums_all = _build_album_groups(store, extras, min_tracks=context_k + 1)
    train_albums, val_albums = _split_albums(albums_all, val_frac=val_frac, seed=seed)
    n_train_tracks = sum(len(v) for v in train_albums.values())
    n_val_tracks = sum(len(v) for v in val_albums.values())
    print(
        f"albums: total={len(albums_all)} (>= {context_k + 1} tracks), "
        f"train={len(train_albums)} ({n_train_tracks} tracks), "
        f"val={len(val_albums)} ({n_val_tracks} tracks)"
    )
    if not train_albums:
        raise click.ClickException("no training albums; lower --context-k or check extras.")

    # Pre-flight: naive retrieval baselines on the same val albums so we
    # have a clear bar the predictor must clear.
    base_centroid = evaluate_baseline(
        matrix,
        val_albums,
        device=torch_device,
        context_k=context_k,
        n_samples_per_album=samples_per_album_eval,
        rng_seed=seed + 9999,
        mode="centroid",
    )
    base_last = evaluate_baseline(
        matrix,
        val_albums,
        device=torch_device,
        context_k=context_k,
        n_samples_per_album=samples_per_album_eval,
        rng_seed=seed + 9999,
        mode="last",
    )
    print(
        f"baseline[centroid]: recall@1={base_centroid['recall@1']:.4f} "
        f"recall@10={base_centroid['recall@10']:.4f} mrr={base_centroid['mrr']:.4f} "
        f"median_rank={base_centroid['median_rank']:.0f}"
    )
    print(
        f"baseline[last]:     recall@1={base_last['recall@1']:.4f} "
        f"recall@10={base_last['recall@10']:.4f} mrr={base_last['mrr']:.4f} "
        f"median_rank={base_last['median_rank']:.0f}"
    )

    cfg = PredictorConfig(context_k=context_k, embedding_dim=embedding_dim)
    model = SessionContextPredictor(cfg).to(torch_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"predictor: {n_params:,} parameters")

    if multi_signal:
        # Build artist groups from the SAME train album rows so we don't
        # leak val tracks into training via the artist source.
        train_row_set = {r for rows in train_albums.values() for r in rows}
        artist_all = _build_artist_groups(store, extras, min_tracks=context_k + 1)
        artist_train = {
            aid: [r for r in rows if r in train_row_set]
            for aid, rows in artist_all.items()
        }
        artist_train = {aid: rows for aid, rows in artist_train.items() if len(rows) >= context_k + 1}
        print(
            f"multi-signal: artist groups (train-only, >={context_k + 1} tracks) = {len(artist_train)}"
        )
        sources = [(train_albums, 1.0), (artist_train, float(artist_weight))]
        train_ds: Dataset = MixedSessionDataset(
            matrix, sources, context_k=context_k, seed=seed
        )
    else:
        train_ds = AlbumSessionDataset(matrix, train_albums, context_k=context_k, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_session,
    )
    n_steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * n_steps_per_epoch

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=100.0,
    )

    best_recall_at_10 = -1.0
    best_report: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pos_cos = 0.0
        n_batches = 0
        for ctx, tgt in train_loader:
            ctx = ctx.to(torch_device)
            tgt = tgt.to(torch_device)
            tgt = F.normalize(tgt, p=2.0, dim=-1)
            pred = model(ctx)  # already normalized
            nce = info_nce_loss(pred, tgt, temperature=temperature)
            cos_to_target = (pred * tgt).sum(dim=-1).mean()
            cos_loss = 1.0 - cos_to_target
            loss = nce + cosine_weight * cos_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            scheduler.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_pos_cos += float(cos_to_target.detach().cpu())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_cos = epoch_pos_cos / max(1, n_batches)

        val_metrics = evaluate(
            model,
            matrix,
            val_albums,
            device=torch_device,
            context_k=context_k,
            n_samples_per_album=samples_per_album_eval,
            rng_seed=seed + epoch,
        )
        line = (
            f"epoch={epoch:02d}/{epochs} "
            f"loss={avg_loss:.4f} train_cos={avg_cos:.4f} "
            f"val_mrr={val_metrics['mrr']:.4f} "
            f"val_recall@10={val_metrics['recall@10']:.4f} "
            f"val_recall@1={val_metrics['recall@1']:.4f} "
            f"median_rank={val_metrics['median_rank']:.0f}"
        )
        print(line, flush=True)

        if val_metrics["recall@10"] > best_recall_at_10:
            best_recall_at_10 = val_metrics["recall@10"]
            best_report = {"epoch": epoch, **val_metrics}
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "predictor_version": PREDICTOR_VERSION,
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg.to_dict(),
                    "store_path": str(store_path),
                    "best_report": best_report,
                },
                out_path,
            )

    final_report = {
        "predictor_version": PREDICTOR_VERSION,
        "n_albums_train": len(train_albums),
        "n_albums_val": len(val_albums),
        "n_params": n_params,
        "best": best_report,
        "baseline_centroid": base_centroid,
        "baseline_last": base_last,
        "config": cfg.to_dict(),
    }
    sys.stdout.write(json.dumps(final_report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
