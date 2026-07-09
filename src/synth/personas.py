"""Persona x goal grid — pre-structured diversity for synthetic sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GOALS = [
    "workout", "focus", "evening_chill", "anime_binge",
    "russian_throwback", "disco_eurodance_party", "film_score_ambient", "discovery",
]

# Session length range per goal (min, max).
_GOAL_LEN = {
    "workout": (8, 20), "focus": (15, 45), "evening_chill": (6, 18),
    "anime_binge": (10, 30), "russian_throwback": (8, 25),
    "disco_eurodance_party": (10, 30), "film_score_ambient": (6, 20), "discovery": (8, 25),
}

# Persona archetypes (activity, conformity, diversity dials).
_ARCHETYPES = [
    ("mainstream_regular", 0.7, 0.8, 0.3),
    ("explorer", 0.6, 0.3, 0.9),
    ("focused_listener", 0.4, 0.6, 0.4),
    ("nostalgic", 0.5, 0.7, 0.35),
    ("power_user", 0.9, 0.5, 0.7),
]

# Default goal weights: over-sample the long-tail to counter head bias.
_DEFAULT_GOAL_WEIGHTS = {
    "workout": 1.0, "focus": 1.2, "evening_chill": 1.0, "anime_binge": 1.5,
    "russian_throwback": 1.5, "disco_eurodance_party": 1.0,
    "film_score_ambient": 0.8, "discovery": 1.5,
}


@dataclass
class Persona:
    name: str
    activity: float
    conformity: float
    diversity: float


@dataclass
class SessionSpec:
    persona: Persona
    goal: str
    session_len: int


def build_grid(
    n_sessions: int,
    rng: np.random.Generator,
    goal_weights: dict[str, float] | None = None,
) -> list[SessionSpec]:
    weights = dict(_DEFAULT_GOAL_WEIGHTS)
    if goal_weights:
        weights.update(goal_weights)
    goals = np.array(GOALS)
    probs = np.array([weights[g] for g in GOALS], dtype=float)
    probs /= probs.sum()
    specs: list[SessionSpec] = []
    for _ in range(n_sessions):
        arc = _ARCHETYPES[rng.integers(0, len(_ARCHETYPES))]
        # small per-session jitter on the dials
        persona = Persona(
            name=arc[0],
            activity=float(np.clip(arc[1] + rng.normal(0, 0.05), 0, 1)),
            conformity=float(np.clip(arc[2] + rng.normal(0, 0.05), 0, 1)),
            diversity=float(np.clip(arc[3] + rng.normal(0, 0.05), 0, 1)),
        )
        goal = str(rng.choice(goals, p=probs))
        lo, hi = _GOAL_LEN[goal]
        specs.append(
            SessionSpec(persona=persona, goal=goal, session_len=int(rng.integers(lo, hi + 1)))
        )
    return specs
