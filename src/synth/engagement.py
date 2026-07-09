"""Engagement-signal model calibrated to real ListeningEventEntity marginals."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

_COMPLETION_THRESHOLD = 0.9  # app defines completed as playedMs >= 0.9 * durationMs


def load_real_events(db_or_csv: str) -> pd.DataFrame:
    if db_or_csv.endswith(".csv"):
        return pd.read_csv(db_or_csv)
    return pd.read_sql_query("SELECT * FROM ListeningEventEntity", sqlite3.connect(db_or_csv))


def calibration_targets(events: pd.DataFrame) -> dict:
    frac = (events["playedMs"] / events["trackDurationMs"]).clip(0, 1)
    completed = events["completed"].astype(bool)
    reasons = events["finalizeReason"].value_counts(normalize=True).to_dict()
    return {
        "completion_rate": float(completed.mean()),
        "finalize_reason": {str(k): float(v) for k, v in reasons.items()},
        "played_fraction_completed": (float(frac[completed].mean()), float(frac[completed].std() or 0.05)),
        "played_fraction_skipped": (float(frac[~completed].mean()), float(frac[~completed].std() or 0.15)),
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
        self._reason_probs = np.array([reasons[r] for r in self._reason_names])
        self._reason_probs = self._reason_probs / self._reason_probs.sum()

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> EngagementModel:
        return cls(calibration_targets(events))

    def _skip_prob(self, pos: int) -> float:
        """Base skip rate, tilted so skips cluster early in a session."""
        base = 1.0 - self._t["completion_rate"]
        tilt = 1.25 if pos < 3 else (0.85 if pos >= 6 else 1.0)
        return float(np.clip(base * tilt, 0.02, 0.98))

    def sample(self, session_len: int, rng: np.random.Generator) -> list[EventLabels]:
        out: list[EventLabels] = []
        for pos in range(session_len):
            skipped = rng.random() < self._skip_prob(pos)
            completed = not skipped
            if completed:
                mean, std = self._t["played_fraction_completed"]
            else:
                mean, std = self._t["played_fraction_skipped"]
            frac = float(np.clip(rng.normal(mean, std), 0.0, 1.0))
            if skipped:
                reason = "USER_SKIPPED"
            else:
                reason = str(rng.choice(self._reason_names, p=self._reason_probs))
                if reason == "USER_SKIPPED":  # completed rows never carry USER_SKIPPED
                    reason = "TRACK_ENDED"
            out.append(EventLabels(frac, skipped, completed, reason))
        return out
