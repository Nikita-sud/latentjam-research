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


def test_kl_finite_on_disjoint_support():
    # A key present in one distribution but absent in the other must not blow up to inf/NaN.
    val = kl_divergence({"a": 1.0}, {"b": 1.0})
    assert np.isfinite(val)


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


def test_hallucinated_ids_do_not_inflate_coverage():
    # Corpus touches only 2 real manifest songs (true coverage 0.2) but is padded with
    # fake ids not in the manifest. The fake ids must NOT count toward coverage.
    man, real = _manifest(), _real_events()
    real_ids = ["s0"] * 200 + ["s1"] * 200
    fake_ids = [f"fake{i}" for i in range(400)]  # hallucinated, absent from manifest
    sessions = pd.DataFrame({"song_id": real_ids + fake_ids, "completed": [1] * 800})
    rep = validate_corpus(sessions, man, real)
    assert not rep.passed
    assert any("coverage" in f for f in rep.failures)
    assert rep.metrics["coverage"] == 0.2  # 2 of 10 manifest songs, not 0.8


def test_missing_completion_signal_fails_safe():
    # An empty / all-NaN `completed` column must FAIL (fail-safe), not silently pass.
    man, real = _manifest(), _real_events()
    rng = np.random.default_rng(2)
    ids = rng.choice(man["song_id"], 800)  # broad coverage, similar genre mix
    sessions = pd.DataFrame({"song_id": ids, "completed": [np.nan] * 800})
    rep = validate_corpus(sessions, man, real)
    assert not rep.passed
    assert any("completion" in f for f in rep.failures)
