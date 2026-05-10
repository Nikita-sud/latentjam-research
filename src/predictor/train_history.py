"""Train HistoryAwarePredictor on synthesized listening events.

Reads the schema produced by ``scripts/synthesize_listening.py`` (the same
schema the Android ``ListeningEventRecorder`` will eventually emit) and
trains the predictor with InfoNCE + cosine anchor.

Held-out users are NOT seen during training, so val measures
generalization to fresh users (the realistic deployment case).

For each event with completed=1 we construct the (history, time, session,
target) tuple as the training pair. Skipped events are still useful as
context (they appear inside other events' history_small) but are not
themselves training targets.
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

from predictor.history_predictor import (
    HISTORY_PREDICTOR_VERSION,
    HistoryAwarePredictor,
    HistoryPredictorConfig,
)
from predictor.store import EmbeddingStore

EXP_DECAY_MEDIUM_DAYS = 30.0
EXP_DECAY_LARGE_DAYS = 365.0
SESSION_GAP_MS = 30 * 60 * 1000.0  # 30 min gap defines a new session

CONTEXT_K = 4


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


def _time_features(ts_unix_ms: float) -> np.ndarray:
    """5-d time encoding: sin/cos hour, sin/cos day-of-week, is_weekend."""
    ts_s = ts_unix_ms / 1000.0
    secs_in_day = (ts_s % (24 * 3600))
    hour = secs_in_day / 3600.0
    days_since_epoch = int(ts_s // (24 * 3600))
    # 1970-01-01 was Thursday (3)
    dow = (days_since_epoch + 3) % 7
    return np.array(
        [
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * dow / 7.0),
            math.cos(2 * math.pi * dow / 7.0),
            float(dow >= 5),
        ],
        dtype=np.float32,
    )


@dataclass
class TrainPair:
    history_small: np.ndarray  # (K, D+1) — last col is played_pct in [0, 1]
    history_medium: np.ndarray  # (D,)
    history_large: np.ndarray   # (D,)
    time_feat: np.ndarray       # (5,)
    session_feat: np.ndarray    # (5,) — last is mean_played_pct
    target: np.ndarray          # (D,)


def _played_pct(played_seconds: float, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return float(np.clip(played_seconds / duration_s, 0.0, 1.0))


def _build_pairs_for_user(
    user_events: pd.DataFrame,
    matrix: np.ndarray,
    track_id_to_row: dict[str, int],
    *,
    context_k: int = CONTEXT_K,
    only_completed_targets: bool = True,
) -> list[TrainPair]:
    """Walk this user's events in time order and emit one TrainPair per event.

    For each event at time t:
    - history_small = embeddings of last K *completed* tracks in same session
                       (zero-padded if fewer); if the current session just
                       started, we pad with this user's medium centroid as
                       a sensible cold-start.
    - history_medium = exp-decay (30 d half-life) centroid of all completed
                        plays *before* time t. If empty, fall back to the
                        first played embedding (cold-start fallback).
    - history_large  = exp-decay (1 y half-life) centroid; same fallback.

    We construct everything causally: the predictor at training time only
    sees what the app would actually have at inference time for this event.
    """
    user_events = user_events.sort_values("ts_unix_ms").reset_index(drop=True)
    pairs: list[TrainPair] = []

    # Streaming centroids (now weighted by played_pct rather than binary
    # completion -- a 95 %-listened track contributes more than a 60 %-one):
    medium_acc = np.zeros(matrix.shape[1], dtype=np.float64)
    medium_w = 0.0
    large_acc = np.zeros(matrix.shape[1], dtype=np.float64)
    large_w = 0.0
    last_event_ts_by_session: dict[str, float] = {}
    played_pct_window: list[float] = []   # last 10 played_pct values
    session_played_pct_history: dict[str, list[float]] = {}
    session_start_ms: dict[str, float] = {}
    last_skipped: int = 0

    # Per-session list of (row, played_pct) in order (used for history_small).
    session_history: dict[str, list[tuple[int, float]]] = {}

    for _, row in user_events.iterrows():
        track_id = row["track_id"]
        if track_id not in track_id_to_row:
            continue
        target_row = track_id_to_row[track_id]
        target_emb = matrix[target_row]

        ts = float(row["ts_unix_ms"])
        sid = str(row["session_id"])
        played_seconds = float(row.get("played_seconds", 0.0))
        track_dur = float(row.get("track_duration_s", 0.0))
        played_pct = _played_pct(played_seconds, track_dur)
        completed = int(row["completed"]) == 1
        skipped = int(row["skipped"]) == 1

        sess_hist = session_history.get(sid, [])
        sess_pct_window = session_played_pct_history.get(sid, [])
        if sid not in session_start_ms:
            session_start_ms[sid] = ts

        # Build history_small (K, D+1): last K tracks in session with played_pct
        # as the extra channel. We include skipped tracks too — the gate value
        # tells the model "this didn't land".
        h_small = np.zeros((context_k, matrix.shape[1] + 1), dtype=np.float32)
        recent = sess_hist[-context_k:]
        if recent:
            stacked = np.stack([matrix[r] for r, _pct in recent], axis=0)
            pcts = np.asarray([pct for _r, pct in recent], dtype=np.float32)
            h_small[-len(recent) :, : matrix.shape[1]] = stacked
            h_small[-len(recent) :, matrix.shape[1]] = pcts
            # Pad with the earliest recent so position embedding still
            # discriminates rather than zero rows.
            if len(recent) < context_k:
                h_small[: context_k - len(recent), : matrix.shape[1]] = stacked[0]
                h_small[: context_k - len(recent), matrix.shape[1]] = pcts[0]
        else:
            if medium_w > 0:
                base = (medium_acc / medium_w).astype(np.float32)
                base = base / (np.linalg.norm(base) + 1e-12)
                h_small[:, : matrix.shape[1]] = base
                h_small[:, matrix.shape[1]] = 0.5  # uncertain played_pct
            else:
                h_small[:, : matrix.shape[1]] = target_emb
                h_small[:, matrix.shape[1]] = 0.5

        # history_medium / history_large from streaming centroids.
        if medium_w > 0:
            h_med = (medium_acc / medium_w).astype(np.float32)
        else:
            h_med = target_emb.copy()
        h_med = h_med / (np.linalg.norm(h_med) + 1e-12)

        if large_w > 0:
            h_lg = (large_acc / large_w).astype(np.float32)
        else:
            h_lg = target_emb.copy()
        h_lg = h_lg / (np.linalg.norm(h_lg) + 1e-12)

        # time + session features.
        time_f = _time_features(ts)

        session_pos = len(sess_hist)
        time_since_session_min = (ts - session_start_ms[sid]) / 60_000.0
        if sess_pct_window:
            mean_pct = float(np.mean(sess_pct_window))
        elif played_pct_window:
            mean_pct = float(np.mean(played_pct_window))
        else:
            mean_pct = 0.5
        # binary completion-rate retained for backwards comparison (>=80%)
        if sess_pct_window:
            comp_rate = float(np.mean([1.0 if v >= 0.8 else 0.0 for v in sess_pct_window]))
        elif played_pct_window:
            comp_rate = float(np.mean([1.0 if v >= 0.8 else 0.0 for v in played_pct_window]))
        else:
            comp_rate = 0.5

        sess_f = np.array(
            [
                math.log1p(session_pos),
                float(last_skipped),
                math.log1p(max(0.0, time_since_session_min)),
                comp_rate,
                mean_pct,
            ],
            dtype=np.float32,
        )

        # Emit pair only if target is completed (we don't train to predict skips).
        if (not only_completed_targets) or completed:
            pairs.append(
                TrainPair(
                    history_small=h_small,
                    history_medium=h_med,
                    history_large=h_lg,
                    time_feat=time_f,
                    session_feat=sess_f,
                    target=target_emb.astype(np.float32),
                )
            )

        # Update streaming state AFTER emitting the pair (so next pair sees
        # this event's contribution). Now we *always* update history_small
        # (so skipped tracks still appear in the K-window with low pct), and
        # weight medium/large centroids by played_pct rather than binary
        # completion -- a 95 %-listened track contributes more than a 60 %-
        # one, and a skipped track still moves the centroid slightly.
        prev_ts = last_event_ts_by_session.get("__GLOBAL__", ts)
        dt_days = max(0.0, (ts - prev_ts) / (24.0 * 3600.0 * 1000.0))
        medium_decay = math.exp(-dt_days * math.log(2) / EXP_DECAY_MEDIUM_DAYS)
        large_decay = math.exp(-dt_days * math.log(2) / EXP_DECAY_LARGE_DAYS)
        medium_acc *= medium_decay
        medium_w *= medium_decay
        large_acc *= large_decay
        large_w *= large_decay
        last_event_ts_by_session["__GLOBAL__"] = ts

        # Reward-shaped weight in [0, 1]: skip (pct < 0.2) contributes ~0,
        # half-listened (~0.5) contributes 0.6, completed (>=0.8) contributes 1.0.
        weight = float(np.clip((played_pct - 0.2) / 0.6, 0.0, 1.0))
        if weight > 0:
            medium_acc += weight * target_emb.astype(np.float64)
            medium_w += weight
            large_acc += weight * target_emb.astype(np.float64)
            large_w += weight

        sess_hist.append((target_row, played_pct))
        session_history[sid] = sess_hist

        comp_window_window_size = 10
        sess_pct_window.append(played_pct)
        if len(sess_pct_window) > comp_window_window_size:
            sess_pct_window.pop(0)
        session_played_pct_history[sid] = sess_pct_window

        played_pct_window.append(played_pct)
        if len(played_pct_window) > 50:
            played_pct_window.pop(0)

        last_skipped = int(skipped)
        last_event_ts_by_session[sid] = ts

    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HistoryEventDataset(Dataset):
    def __init__(self, pairs: list[TrainPair]):
        # Pre-stack into arrays for speed. history_small is (K, D+1), but the
        # centroids and target are pure D-dim embeddings.
        n = len(pairs)
        d_small = pairs[0].history_small.shape[1]   # D+1 when use_played_pct
        d_emb = pairs[0].history_medium.shape[0]    # D
        k = pairs[0].history_small.shape[0]
        self.h_small = np.empty((n, k, d_small), dtype=np.float32)
        self.h_med = np.empty((n, d_emb), dtype=np.float32)
        self.h_lg = np.empty((n, d_emb), dtype=np.float32)
        self.tf = np.empty((n, pairs[0].time_feat.shape[0]), dtype=np.float32)
        self.sf = np.empty((n, pairs[0].session_feat.shape[0]), dtype=np.float32)
        self.tgt = np.empty((n, d_emb), dtype=np.float32)
        for i, p in enumerate(pairs):
            self.h_small[i] = p.history_small
            self.h_med[i] = p.history_medium
            self.h_lg[i] = p.history_large
            self.tf[i] = p.time_feat
            self.sf[i] = p.session_feat
            self.tgt[i] = p.target

    def __len__(self) -> int:
        return self.h_small.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.h_small[idx]),
            torch.from_numpy(self.h_med[idx]),
            torch.from_numpy(self.h_lg[idx]),
            torch.from_numpy(self.tf[idx]),
            torch.from_numpy(self.sf[idx]),
            torch.from_numpy(self.tgt[idx]),
        )


def collate(batch):
    h_small, h_med, h_lg, tf, sf, tgt = zip(*batch, strict=True)
    return (
        torch.stack(list(h_small), dim=0),
        torch.stack(list(h_med), dim=0),
        torch.stack(list(h_lg), dim=0),
        torch.stack(list(tf), dim=0),
        torch.stack(list(sf), dim=0),
        torch.stack(list(tgt), dim=0),
    )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def info_nce(
    pred: torch.Tensor, target: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    pred = F.normalize(pred, p=2.0, dim=-1)
    target = F.normalize(target, p=2.0, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.size(0), device=pred.device)
    loss_p = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_p + loss_t)


@torch.inference_mode()
def evaluate_sampled(
    model: HistoryAwarePredictor,
    val_pairs: list[TrainPair],
    matrix: np.ndarray,
    *,
    device: torch.device,
    n_samples: int = 2000,
    n_negatives: int = 100,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Recall@k measured against (1 positive + n_negatives random) candidates.

    This is the standard recommender metric — it answers "given 100 plausible
    candidates, where does the actual next track rank?" rather than the harder
    "where does it rank in the entire library?". The full-library metric stays
    available via ``evaluate()``.
    """
    rng = np.random.default_rng(rng_seed)
    n_pool = matrix.shape[0]
    if len(val_pairs) > n_samples:
        idxs = rng.choice(len(val_pairs), size=n_samples, replace=False)
    else:
        idxs = np.arange(len(val_pairs))

    pool_t = torch.from_numpy(matrix).to(device)
    model.eval()
    ranks: list[int] = []
    for i in idxs:
        p = val_pairs[i]
        target_row = int(np.argmax(matrix @ p.target))
        # Sample n_negatives negative rows different from target.
        neg_idx = rng.choice(n_pool, size=n_negatives + 5, replace=False)
        neg_idx = neg_idx[neg_idx != target_row][:n_negatives]
        candidate_rows = np.concatenate([[target_row], neg_idx])

        h_small = torch.from_numpy(p.history_small).to(device).unsqueeze(0)
        h_med = torch.from_numpy(p.history_medium).to(device).unsqueeze(0)
        h_lg = torch.from_numpy(p.history_large).to(device).unsqueeze(0)
        tf = torch.from_numpy(p.time_feat).to(device).unsqueeze(0)
        sf = torch.from_numpy(p.session_feat).to(device).unsqueeze(0)
        pred = model(h_small, h_med, h_lg, tf, sf).squeeze(0)  # (D,)

        cand_emb = pool_t[candidate_rows]              # (1+n_neg, D)
        scores = cand_emb @ pred                        # (1+n_neg,)
        order = torch.argsort(scores, descending=True)
        target_pos = int((order == 0).nonzero(as_tuple=False).item())
        ranks.append(target_pos + 1)

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    out: dict[str, float] = {
        "n_samples": int(len(ranks_arr)),
        "mrr": float(np.mean(1.0 / np.maximum(ranks_arr, 1))),
        "median_rank": float(np.median(ranks_arr)),
        "mode": f"sampled-{n_negatives}neg",
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks_arr <= k))
    return out


@torch.inference_mode()
def evaluate(
    model: HistoryAwarePredictor,
    val_pairs: list[TrainPair],
    matrix: np.ndarray,
    *,
    device: torch.device,
    n_samples: int = 2000,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    n_pool = matrix.shape[0]
    pool_t = torch.from_numpy(matrix).to(device)
    model.eval()

    if len(val_pairs) > n_samples:
        idxs = rng.choice(len(val_pairs), size=n_samples, replace=False)
    else:
        idxs = np.arange(len(val_pairs))

    target_rows: list[int] = []
    preds: list[torch.Tensor] = []
    target_id_to_row: dict[bytes, int] = {}
    matrix_keys = matrix.tobytes()  # not used; kept for clarity
    for i in idxs:
        p = val_pairs[i]
        t = p.target
        # find its row by exact-cosine search (target was sourced from matrix
        # so cosine = 1 with one row).
        sims = matrix @ t
        target_row = int(np.argmax(sims))
        target_rows.append(target_row)

        h_small = torch.from_numpy(p.history_small).to(device).unsqueeze(0)
        h_med = torch.from_numpy(p.history_medium).to(device).unsqueeze(0)
        h_lg = torch.from_numpy(p.history_large).to(device).unsqueeze(0)
        tf = torch.from_numpy(p.time_feat).to(device).unsqueeze(0)
        sf = torch.from_numpy(p.session_feat).to(device).unsqueeze(0)
        pred = model(h_small, h_med, h_lg, tf, sf)  # (1, D)
        preds.append(pred)

    pred_t = torch.cat(preds, dim=0)             # (n_eval, D)
    scores = pred_t @ pool_t.T                    # (n_eval, N)
    target_idx = torch.tensor(target_rows, device=device)
    # Exclude history_small rows from each row's candidates? We don't have
    # those handily; in synthetic data the model can still trivially recover
    # them via cosine~1 to history_small. Skip masking for simplicity.

    ranks: list[int] = []
    for i in range(scores.size(0)):
        order = torch.argsort(scores[i], descending=True)
        target_pos = int((order == target_idx[i]).nonzero(as_tuple=False).item())
        ranks.append(target_pos + 1)

    ranks_arr = np.asarray(ranks, dtype=np.int64)
    out: dict[str, float] = {
        "n_samples": int(len(ranks_arr)),
        "mrr": float(np.mean(1.0 / np.maximum(ranks_arr, 1))),
        "median_rank": float(np.median(ranks_arr)),
    }
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks_arr <= k))
    return out


def _baseline_query(p: TrainPair, *, w_small: float, w_med: float, w_lg: float) -> np.ndarray:
    # history_small now has shape (K, D+1) — last col is played_pct, drop it
    # for the cosine baseline. We also use played_pct as a per-track weight so
    # skipped tracks don't pollute the small centroid.
    embeds = p.history_small[:, : p.history_medium.shape[0]]
    pct = p.history_small[:, p.history_medium.shape[0]]
    weights = np.clip((pct - 0.2) / 0.6, 0.0, 1.0).astype(np.float32)
    if weights.sum() > 1e-6:
        small_centroid = (embeds * weights[:, None]).sum(axis=0) / (weights.sum() + 1e-12)
    else:
        small_centroid = embeds.mean(axis=0)
    small_centroid = small_centroid / (np.linalg.norm(small_centroid) + 1e-12)
    q = w_small * small_centroid + w_med * p.history_medium + w_lg * p.history_large
    return q / (np.linalg.norm(q) + 1e-12)


@torch.inference_mode()
def evaluate_centroid_baseline(
    val_pairs: list[TrainPair],
    matrix: np.ndarray,
    *,
    device: torch.device,
    weight_small: float = 0.5,
    weight_medium: float = 0.2,
    weight_large: float = 0.3,
    n_samples: int = 2000,
    rng_seed: int = 0,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    sampled_negatives: int | None = None,
) -> dict[str, float]:
    """Layer-1 baseline: weighted blend of history centroids -> KNN.

    If ``sampled_negatives`` is set, scores against (1 positive + n_neg random
    negatives) instead of the full library — matches ``evaluate_sampled``.
    """
    rng = np.random.default_rng(rng_seed)
    pool_t = torch.from_numpy(matrix).to(device)
    n_pool = matrix.shape[0]
    if len(val_pairs) > n_samples:
        idxs = rng.choice(len(val_pairs), size=n_samples, replace=False)
    else:
        idxs = np.arange(len(val_pairs))
    ranks: list[int] = []
    for i in idxs:
        p = val_pairs[i]
        q = _baseline_query(p, w_small=weight_small, w_med=weight_medium, w_lg=weight_large)
        q_t = torch.from_numpy(q.astype(np.float32)).to(device)
        target_row = int(np.argmax(matrix @ p.target))

        if sampled_negatives is None:
            scores = pool_t @ q_t
            order = torch.argsort(scores, descending=True)
            target_pos = int((order == target_row).nonzero(as_tuple=False).item())
        else:
            neg_idx = rng.choice(n_pool, size=sampled_negatives + 5, replace=False)
            neg_idx = neg_idx[neg_idx != target_row][:sampled_negatives]
            cand_rows = np.concatenate([[target_row], neg_idx])
            cand_emb = pool_t[cand_rows]
            scores = cand_emb @ q_t
            order = torch.argsort(scores, descending=True)
            target_pos = int((order == 0).nonzero(as_tuple=False).item())
        ranks.append(target_pos + 1)
    ranks_arr = np.asarray(ranks, dtype=np.int64)
    out: dict[str, float] = {
        "n_samples": int(len(ranks_arr)),
        "mrr": float(np.mean(1.0 / np.maximum(ranks_arr, 1))),
        "median_rank": float(np.median(ranks_arr)),
        "weights": {"small": weight_small, "medium": weight_medium, "large": weight_large},
    }
    if sampled_negatives is not None:
        out["mode"] = f"sampled-{sampled_negatives}neg"
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(ranks_arr <= k))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--store", "store_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--events", "events_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=512, show_default=True)
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--weight-decay", type=float, default=1e-4, show_default=True)
@click.option("--temperature", type=float, default=0.07, show_default=True)
@click.option("--cosine-weight", type=float, default=0.5, show_default=True)
@click.option("--val-frac-users", type=float, default=0.15, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--limit-events", type=int, default=None)
def main(
    store_path: Path,
    events_path: Path,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    cosine_weight: float,
    val_frac_users: float,
    device: str,
    seed: int,
    limit_events: int | None,
) -> None:
    from student.device import resolve_torch_device

    try:
        torch_device = resolve_torch_device(device)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    store = EmbeddingStore.open(store_path)
    if len(store) == 0:
        raise click.ClickException(f"empty store at {store_path}")
    matrix = store._matrix
    if matrix is None:
        store.rebuild_matrix()
        matrix = store._matrix
    assert matrix is not None
    track_id_to_row = {tid: i for i, tid in enumerate(store.df["track_id"].astype(str).tolist())}

    events = pd.read_parquet(events_path)
    if limit_events is not None:
        events = events.head(limit_events)
    events = events[events["track_id"].astype(str).isin(track_id_to_row)].reset_index(drop=True)
    print(f"events: {len(events)} rows, users: {events['user_id'].nunique()}")

    rng = np.random.default_rng(seed)
    users = sorted(events["user_id"].unique())
    rng.shuffle(users)
    n_val_users = max(1, int(round(len(users) * val_frac_users)))
    val_users = set(users[:n_val_users])
    train_users = set(users[n_val_users:])

    print(f"users: train={len(train_users)} val={len(val_users)}")

    # Build pairs. Walking each user's events independently keeps the
    # streaming centroids isolated.
    train_pairs: list[TrainPair] = []
    val_pairs: list[TrainPair] = []
    for user_id, group in tqdm(events.groupby("user_id"), desc="featurize", unit="user"):
        pairs = _build_pairs_for_user(group, matrix, track_id_to_row)
        if user_id in val_users:
            val_pairs.extend(pairs)
        else:
            train_pairs.extend(pairs)

    print(f"pairs: train={len(train_pairs)} val={len(val_pairs)}")

    # Filter pairs that have at least 2 *real* prior tracks in the session
    # so the predictor isn't just memorizing the cold-start fallback (which
    # leaks the target into history_small for the very first event).
    def _has_session_history(p: TrainPair) -> bool:
        # Cold-start fills history_small with the same vector across the K rows;
        # detect that and exclude the pair to keep the eval honest.
        diffs = np.abs(p.history_small[1:] - p.history_small[:-1]).sum()
        return float(diffs) > 1e-6

    train_pairs_full = [p for p in train_pairs if _has_session_history(p)]
    val_pairs_full = [p for p in val_pairs if _has_session_history(p)]
    print(
        f"after history>=2 filter: train={len(train_pairs_full)} (-{len(train_pairs)-len(train_pairs_full)}) "
        f"val={len(val_pairs_full)} (-{len(val_pairs)-len(val_pairs_full)})"
    )
    train_pairs = train_pairs_full
    val_pairs = val_pairs_full

    # Baseline first — both full-pool and 100-negative sampled.
    base_full = evaluate_centroid_baseline(val_pairs, matrix, device=torch_device, rng_seed=seed)
    base_neg100 = evaluate_centroid_baseline(
        val_pairs, matrix, device=torch_device, rng_seed=seed, sampled_negatives=100
    )
    print(
        f"baseline[full-pool]:  recall@1={base_full['recall@1']:.4f} "
        f"recall@10={base_full['recall@10']:.4f} mrr={base_full['mrr']:.4f} "
        f"median_rank={base_full['median_rank']:.0f}"
    )
    print(
        f"baseline[100-neg]:    recall@1={base_neg100['recall@1']:.4f} "
        f"recall@10={base_neg100['recall@10']:.4f} mrr={base_neg100['mrr']:.4f}"
    )
    base = base_full

    cfg = HistoryPredictorConfig(embedding_dim=int(matrix.shape[1]), context_k=CONTEXT_K)
    model = HistoryAwarePredictor(cfg).to(torch_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"predictor: {n_params:,} parameters")

    train_ds = HistoryEventDataset(train_pairs)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate
    )
    n_steps = max(1, len(train_loader)) * epochs

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps, pct_start=0.05, anneal_strategy="cos",
    )

    best_recall_at_10 = -1.0
    best_report: dict[str, Any] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_pos = 0.0
        n_batches = 0
        for h_small, h_med, h_lg, tf, sf, tgt in train_loader:
            h_small = h_small.to(torch_device)
            h_med = h_med.to(torch_device)
            h_lg = h_lg.to(torch_device)
            tf = tf.to(torch_device)
            sf = sf.to(torch_device)
            tgt = tgt.to(torch_device)
            tgt = F.normalize(tgt, p=2.0, dim=-1)

            pred = model(h_small, h_med, h_lg, tf, sf)
            nce = info_nce(pred, tgt, temperature=temperature)
            cos = (pred * tgt).sum(dim=-1).mean()
            loss = nce + cosine_weight * (1.0 - cos)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            scheduler.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_pos += float(cos.detach().cpu())
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        avg_pos = epoch_pos / max(1, n_batches)
        val_full = evaluate(model, val_pairs, matrix, device=torch_device, rng_seed=seed + epoch)
        val_neg = evaluate_sampled(
            model, val_pairs, matrix, device=torch_device, rng_seed=seed + epoch, n_negatives=100,
        )
        line = (
            f"epoch={epoch:02d}/{epochs} loss={avg_loss:.4f} train_cos={avg_pos:.4f} "
            f"val_full[r@10={val_full['recall@10']:.4f} mrr={val_full['mrr']:.4f}] "
            f"val_neg100[r@1={val_neg['recall@1']:.4f} r@10={val_neg['recall@10']:.4f} mrr={val_neg['mrr']:.4f}]"
        )
        print(line, flush=True)

        val = val_full
        if val["recall@10"] > best_recall_at_10:
            best_recall_at_10 = val["recall@10"]
            best_report = {"epoch": epoch, "val_full": val_full, "val_neg100": val_neg}
            torch.save(
                {
                    "predictor_version": HISTORY_PREDICTOR_VERSION,
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg.to_dict(),
                    "store_path": str(store_path),
                    "events_path": str(events_path),
                    "best_report": best_report,
                },
                out_path,
            )

    final_report = {
        "predictor_version": HISTORY_PREDICTOR_VERSION,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "n_train_users": len(train_users),
        "n_val_users": len(val_users),
        "n_params": n_params,
        "best": best_report,
        "baseline_centroid_full": base_full,
        "baseline_centroid_100neg": base_neg100,
        "config": cfg.to_dict(),
    }
    sys.stdout.write(json.dumps(final_report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
