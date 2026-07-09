"""Corpus validation gate: distribution match + coverage vs the real reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _dist(values: pd.Series, support: list[str]) -> dict:
    counts = values.value_counts().to_dict()
    total = sum(counts.get(k, 0) for k in support) + len(support)  # Laplace smoothing
    return {k: (counts.get(k, 0) + 1) / total for k in support}


def kl_divergence(p: dict, q: dict) -> float:
    support = sorted(set(p) | set(q))
    pa = np.array([p.get(k, 1e-9) for k in support])
    qa = np.array([q.get(k, 1e-9) for k in support])
    pa, qa = pa / pa.sum(), qa / qa.sum()
    return float(np.sum(pa * np.log(pa / qa)))


@dataclass
class ValidationReport:
    passed: bool
    metrics: dict
    failures: list[str]


def validate_corpus(
    sessions: pd.DataFrame,
    manifest: pd.DataFrame,
    real_events: pd.DataFrame,
    *,
    max_genre_kl: float = 0.25,
    min_coverage: float = 0.6,
    max_completion_delta: float = 0.05,
) -> ValidationReport:
    genre_of = manifest.set_index("song_id")["genre"].to_dict()
    support = sorted({g for g in genre_of.values() if isinstance(g, str)})

    syn_genres = sessions["song_id"].map(genre_of).dropna()
    real_genres = real_events["song_id"].map(genre_of).dropna()
    genre_kl = kl_divergence(_dist(syn_genres, support), _dist(real_genres, support))

    coverage = sessions["song_id"].nunique() / max(len(manifest), 1)
    completion_delta = abs(sessions["completed"].mean() - real_events["completed"].mean())

    metrics = {
        "genre_kl": genre_kl,
        "coverage": coverage,
        "completion_delta": completion_delta,
        "n_sessions_events": int(len(sessions)),
    }
    failures = []
    if genre_kl > max_genre_kl:
        failures.append(f"genre_kl {genre_kl:.3f} > {max_genre_kl}")
    if coverage < min_coverage:
        failures.append(f"coverage {coverage:.3f} < {min_coverage}")
    if completion_delta > max_completion_delta:
        failures.append(f"completion_delta {completion_delta:.3f} > {max_completion_delta}")
    return ValidationReport(passed=not failures, metrics=metrics, failures=failures)
