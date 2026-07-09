"""Popularity-bias correction: diverse candidate subsets + post-hoc session reweighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synth.candidates import CandidateTable
from synth.personas import SessionSpec

# Goal -> genres it favors (soft relevance prior; unmatched genres still reachable via the tail).
_GOAL_GENRES = {
    "anime_binge": {"Anime OST", "Game OST"},
    "film_score_ambient": {"Film Score"},
    "russian_throwback": {"Russian Pop", "Russian Rock"},
    "disco_eurodance_party": {"Disco", "Eurodance", "Dance-pop"},
    "workout": {"Hip-Hop", "Hard Rock", "Eurodance"},
}


def candidate_subset(candidates: CandidateTable, spec: SessionSpec, *, k: int, rng) -> list[int]:
    n = len(candidates.song_ids)
    k = min(k, n)
    fav = _GOAL_GENRES.get(spec.goal, set())
    genres = candidates.meta["genre"].fillna("")
    relevant = [i for i in range(n) if genres.iloc[i] in fav]
    rng.shuffle(relevant)
    n_relevant = int(round(k * (1.0 - spec.persona.diversity)))
    picks = list(relevant[:n_relevant])
    picked = set(picks)
    remaining = [i for i in range(n) if i not in picked]
    rng.shuffle(remaining)
    picks += remaining[: k - len(picks)]
    return picks[:k]


def reweight_sessions(
    events: pd.DataFrame,
    *,
    target_freq: dict[str, float] | None,
    rng,
    max_drop: float = 0.5,
) -> pd.DataFrame:
    freq = events["track_id"].value_counts(normalize=True).to_dict()
    if target_freq is None:
        target_freq = {t: 1.0 / len(freq) for t in freq}  # flat head
    # per-session keep score: lower when its tracks are over-represented
    keep_prob: dict[str, float] = {}
    for sid, grp in events.groupby("session_id"):
        over = np.mean(
            [freq.get(t, 0) / max(target_freq.get(t, 1e-9), 1e-9) for t in grp["track_id"]]
        )
        keep_prob[sid] = float(np.clip(1.0 / max(over, 1e-6), 0.05, 1.0))
    session_ids = list(keep_prob)
    order = list(session_ids)
    rng.shuffle(order)
    min_keep = int(np.ceil((1.0 - max_drop) * len(events)))
    kept: set = set()
    total = len(events)
    counts = events.groupby("session_id").size().to_dict()
    for sid in order:
        c = counts[sid]
        # Force-keep if dropping this session would push total below the floor: check the
        # post-drop total (not the pre-drop undecided count), so a single large session can't
        # overshoot past min_keep in one step.
        if rng.random() < keep_prob[sid] or (total - c) < min_keep:
            kept.add(sid)
        else:
            total -= c
    return events[events["session_id"].isin(kept)].reset_index(drop=True)
