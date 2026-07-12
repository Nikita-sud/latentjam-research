import numpy as np

from synth.personas import GOALS, SessionSpec, build_grid


def test_grid_size_and_shape():
    grid = build_grid(500, np.random.default_rng(0))
    assert len(grid) == 500
    assert all(isinstance(s, SessionSpec) for s in grid)
    assert all(s.goal in GOALS for s in grid)
    assert all(2 <= s.session_len <= 60 for s in grid)
    assert all(0.0 <= s.persona.diversity <= 1.0 for s in grid)


def test_grid_is_deterministic_per_seed():
    a = build_grid(50, np.random.default_rng(7))
    b = build_grid(50, np.random.default_rng(7))
    assert [(s.goal, s.session_len, s.persona.name) for s in a] == \
           [(s.goal, s.session_len, s.persona.name) for s in b]


def test_goal_weights_bias_the_mix():
    grid = build_grid(2000, np.random.default_rng(1), goal_weights={"discovery": 10.0})
    frac = sum(s.goal == "discovery" for s in grid) / len(grid)
    assert frac > 0.4  # heavily up-weighted goal dominates
