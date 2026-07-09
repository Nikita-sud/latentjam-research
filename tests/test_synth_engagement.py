import numpy as np
import pandas as pd

from synth.engagement import EngagementModel, calibration_targets


def _fake_events(n=4000, completion=0.44, seed=0):
    rng = np.random.default_rng(seed)
    completed = rng.random(n) < completion
    return pd.DataFrame(
        {
            "completed": completed.astype(int),
            "skipped": (~completed).astype(int),
            "playedMs": np.where(completed, 200000, rng.integers(1000, 60000, n)),
            "trackDurationMs": 210000,
            "sessionPos": rng.integers(0, 15, n),
            "finalizeReason": rng.choice(
                ["TRACK_ENDED", "NEW_PLAYBACK", "USER_SKIPPED", "SESSION_END"],
                n, p=[0.42, 0.32, 0.25, 0.01],
            ),
        }
    )


def _corpus(model, session_len, n_sessions, rng):
    return [lbl for _ in range(n_sessions) for lbl in model.sample(session_len=session_len, rng=rng)]


def test_calibration_targets_match_input():
    ev = _fake_events()
    t = calibration_targets(ev)
    assert abs(t["completion_rate"] - 0.44) < 0.03
    assert set(t["finalize_reason"]) == {"TRACK_ENDED", "NEW_PLAYBACK", "USER_SKIPPED", "SESSION_END"}
    assert abs(sum(t["finalize_reason"].values()) - 1.0) < 1e-6


def test_sampled_corpus_reproduces_completion_rate():
    ev = _fake_events()
    model = EngagementModel.from_events(ev)
    rng = np.random.default_rng(1)
    labels = _corpus(model, 10, 500, rng)
    got = np.mean([lbl.completed for lbl in labels])
    assert abs(got - 0.44) < 0.04
    # completed rows carry a high played fraction; skipped rows a low one
    comp = [lbl.played_fraction for lbl in labels if lbl.completed]
    skip = [lbl.played_fraction for lbl in labels if lbl.skipped]
    assert np.mean(comp) > 0.85 and np.mean(skip) < 0.5
    # every label is internally consistent
    assert all(lbl.completed != lbl.skipped for lbl in labels)


def test_completion_rate_invariant_to_session_len():
    # The position tilt must be normalized so the reproduced completion rate does
    # not drift with session length (Fix 1).
    ev = _fake_events()
    model = EngagementModel.from_events(ev)
    for session_len in (3, 10, 50):
        rng = np.random.default_rng(session_len)
        labels = _corpus(model, session_len, 2000, rng)
        got = np.mean([lbl.completed for lbl in labels])
        assert abs(got - 0.44) < 0.04, f"session_len={session_len}: completion {got:.3f}"


def test_played_fraction_respects_completion_threshold():
    # Every completed row >= 0.9, every skipped row < 0.9 (Fix 2), for EVERY event.
    ev = _fake_events()
    model = EngagementModel.from_events(ev)
    rng = np.random.default_rng(7)
    labels = _corpus(model, 10, 500, rng)
    assert all(lbl.played_fraction >= 0.9 for lbl in labels if lbl.completed)
    assert all(lbl.played_fraction < 0.9 for lbl in labels if lbl.skipped)


def test_no_completed_label_has_user_skipped_reason():
    # Completed rows never carry USER_SKIPPED (Fix 5 / structural invariant).
    ev = _fake_events()
    model = EngagementModel.from_events(ev)
    rng = np.random.default_rng(11)
    labels = _corpus(model, 10, 500, rng)
    assert not any(lbl.finalize_reason == "USER_SKIPPED" for lbl in labels if lbl.completed)


def test_degenerate_all_completed_input_yields_finite_fractions():
    # An all-completed calibration bucket makes the skipped mean/std NaN; a plain
    # `or` fallback misses NaN (NaN is truthy) and NaN propagates through
    # rng.normal. The skip-prob floor (0.02) still samples the occasional skipped
    # row, so both buckets must be NaN-safe (Fix 3).
    n = 500
    ev = pd.DataFrame(
        {
            "completed": np.ones(n, dtype=int),
            "skipped": np.zeros(n, dtype=int),
            "playedMs": 200000,
            "trackDurationMs": 210000,
            "sessionPos": np.arange(n) % 15,
            "finalizeReason": "TRACK_ENDED",
        }
    )
    t = calibration_targets(ev)
    assert np.isfinite(t["played_fraction_completed"]).all()
    assert np.isfinite(t["played_fraction_skipped"]).all()

    model = EngagementModel.from_events(ev)
    rng = np.random.default_rng(3)
    labels = _corpus(model, 10, 200, rng)
    assert all(np.isfinite(lbl.played_fraction) for lbl in labels)
