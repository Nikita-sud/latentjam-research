import numpy as np
import pandas as pd

from synth.validate import kl_divergence, validate_corpus


def _manifest():
    genres = ["Anime OST"] * 5 + ["Hip-Hop"] * 3 + ["Disco"] * 2
    return pd.DataFrame({"song_id": [f"s{i}" for i in range(10)], "genre": genres})


def _real_events():
    # ~44% completion, genre freq roughly matching the manifest
    rng = np.random.default_rng(0)
    ids = rng.choice([f"s{i}" for i in range(10)], 400)
    return pd.DataFrame({"song_id": ids, "completed": (rng.random(400) < 0.44).astype(int)})


def test_kl_zero_for_identical():
    assert kl_divergence({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) < 1e-9


def test_good_corpus_passes():
    man, real = _manifest(), _real_events()
    rng = np.random.default_rng(1)
    ids = rng.choice(man["song_id"], 800)  # broad coverage, similar genre mix
    sessions = pd.DataFrame({"song_id": ids, "completed": (rng.random(800) < 0.44).astype(int)})
    rep = validate_corpus(sessions, man, real)
    assert rep.passed, rep.failures


def test_collapsed_corpus_fails_on_coverage_and_kl():
    man, real = _manifest(), _real_events()
    sessions = pd.DataFrame({"song_id": ["s0"] * 800, "completed": [1] * 800})  # mode collapse
    rep = validate_corpus(sessions, man, real)
    assert not rep.passed
    assert any("coverage" in f for f in rep.failures)
