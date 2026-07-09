"""Engagement-signal model calibrated to real ListeningEventEntity marginals."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

_COMPLETION_THRESHOLD = 0.9  # app defines completed as playedMs >= 0.9 * durationMs
_SKIP_PROB_CLIP = (0.02, 0.98)


def load_real_events(db_or_csv: str) -> pd.DataFrame:
    if db_or_csv.endswith(".csv"):
        return pd.read_csv(db_or_csv)
    return pd.read_sql_query("SELECT * FROM ListeningEventEntity", sqlite3.connect(db_or_csv))


def _frac_summary(frac: pd.Series, default_mean: float, default_std: float) -> tuple[float, float]:
    """Robust (mean, std) for a played-fraction bucket.

    A degenerate bucket (0 or 1 rows, e.g. all-completed calibration data) yields
    a NaN mean and/or NaN std; NaN is truthy so a plain ``or`` fallback misses it.
    Fall back explicitly on any non-finite (or zero-std) value so nothing NaN can
    reach ``rng.normal`` at sample time.
    """
    mean = frac.mean()
    std = frac.std()
    safe_mean = default_mean if not np.isfinite(mean) else float(mean)
    safe_std = default_std if not np.isfinite(std) or std == 0 else float(std)
    return safe_mean, safe_std


def calibration_targets(events: pd.DataFrame) -> dict:
    frac = (events["playedMs"] / events["trackDurationMs"]).clip(0, 1)
    completed = events["completed"].astype(bool)
    reasons = events["finalizeReason"].value_counts(normalize=True).to_dict()
    return {
        "completion_rate": float(completed.mean()),
        "finalize_reason": {str(k): float(v) for k, v in reasons.items()},
        "played_fraction_completed": _frac_summary(frac[completed], 0.95, 0.05),
        "played_fraction_skipped": _frac_summary(frac[~completed], 0.2, 0.15),
    }


@dataclass
class EventLabels:
    played_fraction: float
    skipped: bool
    completed: bool
    finalize_reason: str


class EngagementModel:
    def __init__(self, targets: dict):
        self._t = targets
        reasons = targets["finalize_reason"]
        self._reason_names = list(reasons)
        self._reason_probs = np.array([reasons[r] for r in self._reason_names], dtype=float)
        self._reason_probs = self._reason_probs / self._reason_probs.sum()
        # Completed rows never carry USER_SKIPPED: sample them from the remaining
        # reasons, renormalized, so the skipped mass is redistributed proportionally
        # instead of dumped entirely onto TRACK_ENDED.
        comp_names = [r for r in self._reason_names if r != "USER_SKIPPED"]
        comp_probs = np.array([reasons[r] for r in comp_names], dtype=float)
        if comp_probs.sum() > 0:
            self._completed_reason_names = comp_names
            self._completed_reason_probs = comp_probs / comp_probs.sum()
        else:  # degenerate: every observed reason was USER_SKIPPED
            self._completed_reason_names = ["TRACK_ENDED"]
            self._completed_reason_probs = np.array([1.0])

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> EngagementModel:
        return cls(calibration_targets(events))

    @staticmethod
    def _position_tilt(pos: int) -> float:
        """Raw skip-rate tilt so skips cluster early in a session."""
        return 1.25 if pos < 3 else (0.85 if pos >= 6 else 1.0)

    def _skip_probs(self, session_len: int) -> np.ndarray:
        """Per-position skip probabilities for a session of length ``session_len``.

        The position tilt is applied as WEIGHTS normalized to mean 1.0 over the
        session, so the expected skip probability averaged over the session equals
        the base skip rate for any ``session_len`` (up to the safety clip). This
        keeps the reproduced completion rate invariant to session length.
        """
        if session_len <= 0:
            return np.empty(0)
        base = 1.0 - self._t["completion_rate"]
        tilts = np.array([self._position_tilt(p) for p in range(session_len)], dtype=float)
        weights = tilts / tilts.mean()  # mean 1.0 -> preserves the base skip rate
        return np.clip(base * weights, *_SKIP_PROB_CLIP)

    def sample(self, session_len: int, rng: np.random.Generator) -> list[EventLabels]:
        out: list[EventLabels] = []
        skip_probs = self._skip_probs(session_len)
        for pos in range(session_len):
            skipped = rng.random() < skip_probs[pos]
            completed = not skipped
            if completed:
                mean, std = self._t["played_fraction_completed"]
                # Structurally enforce the app's completed definition.
                frac = float(np.clip(rng.normal(mean, std), _COMPLETION_THRESHOLD, 1.0))
                reason = str(rng.choice(self._completed_reason_names, p=self._completed_reason_probs))
            else:
                mean, std = self._t["played_fraction_skipped"]
                frac = float(np.clip(rng.normal(mean, std), 0.0, _COMPLETION_THRESHOLD - 1e-6))
                reason = "USER_SKIPPED"
            out.append(EventLabels(frac, skipped, completed, reason))
        return out
