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

    # Coverage counts only manifest-present ids: hallucinated ids not in the manifest
    # must never inflate coverage (else a mode-collapsed corpus padded with fake ids passes).
    manifest_ids = set(manifest["song_id"])
    covered = sessions["song_id"][sessions["song_id"].isin(manifest_ids)].nunique()
    # Denominator is the count of *distinct* manifest ids, not row count: duplicate
    # song_id rows in the manifest must not cap achievable coverage below 1.0.
    coverage = covered / max(manifest["song_id"].nunique(), 1)

    failures = []
    if genre_kl > max_genre_kl:
        failures.append(f"genre_kl {genre_kl:.3f} > {max_genre_kl}")
    if coverage < min_coverage:
        failures.append(f"coverage {coverage:.3f} < {min_coverage}")

    # Missing/NaN completion signal must fail safe (FAIL), not fail open. An empty or all-NaN
    # `completed` column makes `.mean()` NaN and `NaN > threshold` False, silently skipping the
    # check; treat it as a failure instead.
    syn_mean = sessions["completed"].mean()
    real_mean = real_events["completed"].mean()
    if not np.isfinite(syn_mean) or not np.isfinite(real_mean):
        failures.append("completion signal missing or NaN")
        completion_delta = float("nan")
    else:
        completion_delta = abs(syn_mean - real_mean)
        if completion_delta > max_completion_delta:
            failures.append(f"completion_delta {completion_delta:.3f} > {max_completion_delta}")

    metrics = {
        "genre_kl": genre_kl,
        "coverage": coverage,
        "completion_delta": completion_delta,
        "n_sessions_events": int(len(sessions)),
    }
    return ValidationReport(passed=not failures, metrics=metrics, failures=failures)
