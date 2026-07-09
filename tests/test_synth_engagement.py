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
    labels = [lbl for _ in range(500) for lbl in model.sample(session_len=10, rng=rng)]
    got = np.mean([lbl.completed for lbl in labels])
    assert abs(got - 0.44) < 0.04
    # completed rows carry a high played fraction; skipped rows a low one
    comp = [lbl.played_fraction for lbl in labels if lbl.completed]
    skip = [lbl.played_fraction for lbl in labels if lbl.skipped]
    assert np.mean(comp) > 0.85 and np.mean(skip) < 0.5
    # every label is internally consistent
    assert all(lbl.completed != lbl.skipped for lbl in labels)
