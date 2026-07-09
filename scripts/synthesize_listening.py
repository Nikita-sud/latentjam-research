"""Synthesize realistic listening-event sequences over an EmbeddingStore.

We don't have any real LatentJam user history yet (the Android app's
``ListeningEventRecorder`` is v0.5+), and there's no public dataset that
gives us aligned (history, time-of-day, session, next-track) tuples on
the same audio embeddings the app will use. So we **simulate**: fake
users with stable taste centroids, time-of-day energy patterns, mood
drift inside a session, and skip behavior based on track-vs-taste fit.

The simulation generates the exact schema the Android app's
``ListeningEvent`` will eventually produce, so the predictor we train on
the synthetic stream can be **fine-tuned** on real data without code
changes when listening logs accumulate. That's the whole point of
synthesizing rather than using album proxies — the *shape* of the
training data matches reality, even if the *values* are fictional.

What we encode in the simulation:

- Per-user taste centroid in the same 512-d store space (sampled from
  the store manifold, anchored on a primary genre).
- Per-user energy-by-hour curve (e.g. high-energy mornings, mellow
  evenings, drawn from a few archetypes + noise).
- Mood drift inside a session (an EMA of the recent context steers the
  next-track sampling).
- Session length distributions (Poisson per day, NegBin per session).
- Skip / completion based on the cosine between candidate and the user's
  current "want vector" (taste + drift + hour energy).

The output is a parquet with one row per played track, in the schema
``listening_event``:

    user_id, session_id, ts_unix_ms, track_id, played_seconds,
    track_duration_s, completed, skipped, liked,
    context_track_ids (list<string>, last 4 prior tracks in session),
    hour_of_day, day_of_week, is_weekend
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
from tqdm import tqdm

from predictor.store import EmbeddingStore

EMBED_DIM = 512  # validated against the store at load time
TRACK_DURATION_DEFAULT_S = 30.0  # FMA-medium clip length; real app uses actual track duration
CONTEXT_K = 4


@dataclass
class UserProfile:
    """Stable per-user state that drives the simulation."""

    user_id: str
    taste_centroid: np.ndarray  # (D,), L2-normalized
    taste_spread: float          # std of cosine acceptance (skip threshold ~ 1 - 2*spread)
    energy_by_hour: np.ndarray   # (24,), mean 1.0; >1 = wants high-energy at that hour
    sessions_per_day: float       # Poisson rate
    session_length_mean: float    # tracks per session (NegBin mean)
    session_length_overdisp: float
    skip_floor: float             # absolute cosine below this -> skip ~always
    novelty_pref: float           # 0=replay-heavy, 1=always-new
    drift_weight: float           # 0=stick to taste, 1=fully follow recent context

    def hour_energy(self, hour: int) -> float:
        return float(self.energy_by_hour[hour % 24])


# ---------------------------------------------------------------------------
# Track features derived from the store
# ---------------------------------------------------------------------------


@dataclass
class TrackTable:
    matrix: np.ndarray              # (N, D) L2-normalized
    track_ids: list[str]
    energy: np.ndarray              # (N,) ~ proxy energy per track in [0, 1]
    genre: list[str | None]
    duration_s: np.ndarray          # (N,)


def _track_energy_proxy(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """We don't have explicit energy — derive a stable proxy from the embedding.

    Project onto a fixed random direction in latent space, sigmoid-squash to
    [0, 1]. This is reproducible across users (same direction) and only
    requires the embedding matrix, no audio re-decoding.
    """
    direction = rng.standard_normal(matrix.shape[1]).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-12
    raw = matrix @ direction
    return 1.0 / (1.0 + np.exp(-raw * 4.0))


def build_track_table(store: EmbeddingStore, *, energy_seed: int = 0) -> TrackTable:
    matrix = store._matrix
    if matrix is None:
        store.rebuild_matrix()
        matrix = store._matrix
    assert matrix is not None
    df = store.df
    track_ids = df["track_id"].astype(str).tolist()
    genre = [g if isinstance(g, str) else None for g in df["genre"].tolist()]
    rng = np.random.default_rng(energy_seed)
    energy = _track_energy_proxy(matrix, rng)
    duration_s = np.full((matrix.shape[0],), TRACK_DURATION_DEFAULT_S, dtype=np.float32)
    return TrackTable(matrix=matrix, track_ids=track_ids, energy=energy, genre=genre, duration_s=duration_s)


# ---------------------------------------------------------------------------
# User sampling
# ---------------------------------------------------------------------------


_HOUR_ARCHETYPES = {
    # 24-hour energy curves (mean ~1.0). Sampled stochastically per user with noise.
    "morning_workout": np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 1.4, 1.7, 1.6, 1.4, 1.2,
                                 1.0, 0.9, 0.9, 0.9, 1.0, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5]),
    "office_drone":    np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8, 1.0, 1.1, 1.2, 1.2,
                                 1.2, 1.1, 1.2, 1.2, 1.1, 1.0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    "night_owl":       np.array([1.4, 1.3, 1.0, 0.7, 0.4, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7,
                                 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5, 1.5, 1.4]),
    "balanced":        np.array([0.6, 0.5, 0.5, 0.5, 0.5, 0.7, 0.9, 1.0, 1.1, 1.1, 1.1,
                                 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 0.9, 0.8, 0.7]),
    "evening_chill":   np.array([0.5, 0.5, 0.4, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9,
                                 0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.3, 1.5, 1.6, 1.5, 1.3, 1.0, 0.8]),
}


def _sample_user(
    user_idx: int, table: TrackTable, rng: np.random.Generator
) -> UserProfile:
    # Anchor taste centroid on a random track + small spread direction.
    anchor_row = int(rng.integers(0, table.matrix.shape[0]))
    base = table.matrix[anchor_row].copy()
    perturb = rng.standard_normal(base.shape[0]).astype(np.float32) * 0.15
    centroid = base + perturb
    centroid /= np.linalg.norm(centroid) + 1e-12

    archetype_name = rng.choice(list(_HOUR_ARCHETYPES.keys()))
    energy_curve = _HOUR_ARCHETYPES[archetype_name].copy()
    energy_curve += rng.normal(0.0, 0.15, size=24)
    energy_curve = np.clip(energy_curve, 0.1, 2.5)

    return UserProfile(
        user_id=f"u{user_idx:04d}",
        taste_centroid=centroid.astype(np.float32),
        taste_spread=float(rng.uniform(0.12, 0.22)),
        energy_by_hour=energy_curve.astype(np.float32),
        sessions_per_day=float(rng.uniform(0.6, 3.0)),
        session_length_mean=float(rng.uniform(4.0, 14.0)),
        session_length_overdisp=float(rng.uniform(1.5, 4.0)),
        skip_floor=float(rng.uniform(0.05, 0.20)),
        novelty_pref=float(rng.uniform(0.4, 0.95)),
        drift_weight=float(rng.uniform(0.15, 0.55)),
    )


# ---------------------------------------------------------------------------
# Session simulation
# ---------------------------------------------------------------------------


def _candidate_sample(
    user: UserProfile,
    drift: np.ndarray,
    table: TrackTable,
    *,
    hour_energy_target: float,
    play_counts: dict[int, int],
    last_played_at: dict[int, float],
    now_ts_ms: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> int:
    """Pick a row index in the store as the next track.

    We don't enumerate all 25k rows — we sample n_candidates and softmax over
    them by fit score. This both speeds up simulation and prevents the same
    track from dominating every step.
    """
    # Want vector = taste + drift_weight * recent_drift, re-normalized.
    want = user.taste_centroid + user.drift_weight * drift
    want_norm = np.linalg.norm(want) + 1e-12
    want = want / want_norm

    # Sample candidates with bias toward "near want" for efficiency.
    # 70% near (top-K nearest by want), 30% random for diversity.
    n_near = int(n_candidates * 0.7)
    n_rand = n_candidates - n_near
    near_scores = table.matrix @ want
    # take top 200 to sample from to avoid full sort
    top_pool_size = 500
    pool = np.argpartition(-near_scores, top_pool_size)[:top_pool_size]
    near_idx = rng.choice(pool, size=n_near, replace=False)
    rand_idx = rng.integers(0, table.matrix.shape[0], size=n_rand)
    candidates = np.unique(np.concatenate([near_idx, rand_idx]))

    cand_emb = table.matrix[candidates]
    cand_energy = table.energy[candidates]

    # Scoring components (all in same scale-ish):
    fit = cand_emb @ want                              # cosine to want
    energy_match = -np.abs(cand_energy - hour_energy_target * 0.5 - 0.25)  # 0 best
    energy_match = energy_match * 1.5
    novelty = np.array(
        [
            1.0 / (1.0 + play_counts.get(int(c), 0))
            for c in candidates
        ],
        dtype=np.float32,
    )
    recency_penalty = np.array(
        [
            -3.0 if (now_ts_ms - last_played_at.get(int(c), -1e15)) < 6 * 3600 * 1000 else 0.0
            for c in candidates
        ],
        dtype=np.float32,
    )

    score = (
        4.0 * fit
        + 0.5 * energy_match
        + 0.5 * user.novelty_pref * np.log(novelty + 1e-3)
        + recency_penalty
    )
    # Softmax sample with sharper temperature so the target is recoverable.
    score = score - score.max()
    probs = np.exp(score / 0.2)
    probs /= probs.sum() + 1e-12
    chosen = int(rng.choice(candidates, p=probs))
    return chosen


def _simulate_user(
    user: UserProfile,
    table: TrackTable,
    *,
    n_days: int,
    start_ts_ms: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    play_counts: dict[int, int] = {}
    last_played_at: dict[int, float] = {}

    DAY_MS = 24 * 3600 * 1000.0
    HOUR_MS = 3600 * 1000.0

    for day in range(n_days):
        n_sessions = rng.poisson(user.sessions_per_day)
        n_sessions = int(np.clip(n_sessions, 0, 6))
        for s in range(n_sessions):
            # Sample session start hour weighted by user's energy curve.
            hour_probs = user.energy_by_hour / user.energy_by_hour.sum()
            start_hour = int(rng.choice(24, p=hour_probs))
            start_minute = int(rng.integers(0, 60))
            session_start_ms = start_ts_ms + day * DAY_MS + start_hour * HOUR_MS + start_minute * 60_000

            # Session length from negative-binomial-ish (Poisson with overdispersion).
            length = max(2, int(rng.poisson(user.session_length_mean) + rng.poisson(user.session_length_overdisp)))
            length = min(length, 30)

            session_id = f"{user.user_id}-d{day:03d}-s{s}"
            drift = np.zeros(table.matrix.shape[1], dtype=np.float32)
            context_window: list[str] = []
            cur_ts = session_start_ms

            for pos in range(length):
                hour_now = int(((cur_ts - start_ts_ms) // HOUR_MS) % 24)
                hour_energy = user.hour_energy(hour_now)
                row = _candidate_sample(
                    user,
                    drift,
                    table,
                    hour_energy_target=float(hour_energy),
                    play_counts=play_counts,
                    last_played_at=last_played_at,
                    now_ts_ms=cur_ts,
                    n_candidates=40,
                    rng=rng,
                )
                track_emb = table.matrix[row]

                # Decide skip / complete based on fit vs user's want vector.
                want = user.taste_centroid + user.drift_weight * drift
                want = want / (np.linalg.norm(want) + 1e-12)
                fit = float(track_emb @ want)
                # Hour-energy match also influences skip prob (mismatch -> skip more).
                energy_mismatch = abs(table.energy[row] - hour_energy * 0.5 - 0.25)
                p_skip = (
                    np.clip(1.0 - (fit - user.skip_floor) / max(1e-6, user.taste_spread), 0.0, 0.95) * 0.5
                    + 0.4 * energy_mismatch
                    + 0.05  # baseline noise
                )
                p_skip = float(np.clip(p_skip, 0.02, 0.85))
                skipped = bool(rng.random() < p_skip)

                if skipped:
                    played_seconds = float(rng.uniform(2.0, 25.0))
                    completed = 0
                    # Skip events don't update drift much (user disliked it).
                    drift = 0.95 * drift
                else:
                    played_seconds = float(table.duration_s[row]) * float(rng.uniform(0.8, 1.0))
                    completed = 1
                    # Update drift toward this track.
                    drift = 0.85 * drift + 0.15 * (track_emb - user.taste_centroid)

                liked = int(not skipped and fit > user.skip_floor + user.taste_spread and rng.random() < 0.05)

                events.append(
                    {
                        "user_id": user.user_id,
                        "session_id": session_id,
                        "ts_unix_ms": int(cur_ts),
                        "track_id": table.track_ids[row],
                        "track_row": int(row),  # debug; useful for fast lookups in training
                        "played_seconds": float(played_seconds),
                        "track_duration_s": float(table.duration_s[row]),
                        "completed": int(completed),
                        "skipped": int(skipped),
                        "liked": int(liked),
                        "context_track_ids": list(context_window[-CONTEXT_K:]),
                        "hour_of_day": int(hour_now),
                        "day_of_week": int(((day + 0) % 7)),  # arbitrary anchor
                        "is_weekend": int(((day + 0) % 7) >= 5),
                        "session_pos": int(pos),
                    }
                )
                play_counts[row] = play_counts.get(row, 0) + 1
                last_played_at[row] = cur_ts
                context_window.append(table.track_ids[row])
                cur_ts += float(played_seconds * 1000.0)

                # Skip-then-skip-fast often ends a session.
                if skipped and rng.random() < 0.3:
                    break

    return events


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--store",
    "store_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--n-users", type=int, default=200, show_default=True)
@click.option("--days-per-user", type=int, default=30, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
def main(
    store_path: Path, out_path: Path, n_users: int, days_per_user: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    store = EmbeddingStore.open(store_path)
    if len(store) == 0:
        raise click.ClickException(f"empty store at {store_path}")
    table = build_track_table(store, energy_seed=seed)
    print(f"store: {len(store)} tracks, dim={table.matrix.shape[1]}")
    print(f"simulating {n_users} users x {days_per_user} days")

    start_ts_ms = float((pd.Timestamp("2026-01-01", tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds() * 1000)

    all_events: list[dict[str, Any]] = []
    for u in tqdm(range(n_users), desc="users"):
        user = _sample_user(u, table, rng)
        evts = _simulate_user(user, table, n_days=days_per_user, start_ts_ms=start_ts_ms, rng=rng)
        all_events.extend(evts)

    df = pd.DataFrame(all_events)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    completed = int(df["completed"].sum())
    skipped = int(df["skipped"].sum())
    liked = int(df["liked"].sum())
    n_sessions = df["session_id"].nunique()

    report = {
        "out": str(out_path),
        "n_users": int(n_users),
        "n_events": int(len(df)),
        "n_sessions": int(n_sessions),
        "n_completed": completed,
        "n_skipped": skipped,
        "n_liked": liked,
        "completion_rate": completed / max(1, len(df)),
        "skip_rate": skipped / max(1, len(df)),
        "events_per_user_p50": float(df.groupby("user_id").size().median()),
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
