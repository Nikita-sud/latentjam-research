"""Corpus driver: pure generate_corpus pipeline with the LLM call injected."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from synth.candidates import CandidateTable
from synth.engagement import EngagementModel
from synth.personas import build_grid

_mod_path = Path(__file__).resolve().parents[1] / "scripts" / "synth_generate_corpus.py"
spec = importlib.util.spec_from_file_location("synth_generate_corpus", _mod_path)
driver = importlib.util.module_from_spec(spec)
sys.modules["synth_generate_corpus"] = driver
spec.loader.exec_module(driver)


def _ct(n=40):
    meta = pd.DataFrame({"song_id":[f"uas-{i}" for i in range(n)], "title":[f"T{i}" for i in range(n)],
                         "artist":["a"]*n, "album":[None]*n, "genre":["Pop"]*n, "year":[2000]*n,
                         "bpm":[120.0]*n, "energy":[0.5]*n})
    return CandidateTable(list(meta["song_id"]), {s:i for i,s in enumerate(meta["song_id"])},
                          np.zeros((n,960),dtype=np.float32), meta)


def _eng():
    ev = pd.DataFrame({"completed":[1,0]*100,"skipped":[0,1]*100,"playedMs":[200000,20000]*100,
                       "trackDurationMs":[210000]*200,"finalizeReason":["TRACK_ENDED","USER_SKIPPED"]*100})
    return EngagementModel.from_events(ev)


def test_generate_corpus_produces_export_schema():
    ct, eng = _ct(), _eng()
    grid = build_grid(20, np.random.default_rng(0))
    # fake LLM: pick the first min(session_len, len(subset)) of the subset in order
    def fake_gen(spec_, cand, subset):
        return [cand.song_ids[i] for i in subset[: spec_.session_len]]
    df = driver.generate_corpus(ct, eng, grid, model="x", k=20, rng=np.random.default_rng(0), generate_fn=fake_gen)
    assert set(df.columns) == {"user_id","session_id","ts_unix_ms","track_id","track_row","played_seconds",
        "track_duration_s","completed","skipped","liked","context_track_ids","hour_of_day",
        "day_of_week","is_weekend","session_pos"}
    assert df["session_id"].nunique() == 20
    assert set(df["track_id"]).issubset(set(ct.song_ids))  # no hallucinated ids


def test_generate_corpus_skips_session_when_generate_fn_raises():
    ct, eng = _ct(), _eng()
    grid = build_grid(5, np.random.default_rng(1))

    def flaky_gen(spec_, cand, subset):
        if flaky_gen.calls == 0:
            flaky_gen.calls += 1
            raise ValueError("malformed JSON from LLM")
        return [cand.song_ids[i] for i in subset[: spec_.session_len]]
    flaky_gen.calls = 0

    df = driver.generate_corpus(ct, eng, grid, model="x", k=20, rng=np.random.default_rng(1), generate_fn=flaky_gen)
    assert df["session_id"].nunique() == 4  # one session dropped, run continues


def test_generate_corpus_skips_session_on_empty_generation():
    ct, eng = _ct(), _eng()
    grid = build_grid(5, np.random.default_rng(2))

    def empty_then_ok(spec_, cand, subset):
        if empty_then_ok.calls == 0:
            empty_then_ok.calls += 1
            return []
        return [cand.song_ids[i] for i in subset[: spec_.session_len]]
    empty_then_ok.calls = 0

    df = driver.generate_corpus(ct, eng, grid, model="x", k=20, rng=np.random.default_rng(2),
                                 generate_fn=empty_then_ok)
    assert df["session_id"].nunique() == 4


def test_generate_corpus_skips_session_on_hallucinated_song_id():
    ct, eng = _ct(), _eng()
    grid = build_grid(3, np.random.default_rng(3))

    def hallucinate_first(spec_, cand, subset):
        if hallucinate_first.calls == 0:
            hallucinate_first.calls += 1
            return ["not-a-real-song-id"]
        return [cand.song_ids[i] for i in subset[: spec_.session_len]]
    hallucinate_first.calls = 0

    df = driver.generate_corpus(ct, eng, grid, model="x", k=20, rng=np.random.default_rng(3),
                                 generate_fn=hallucinate_first)
    assert df["session_id"].nunique() == 2
    assert set(df["track_id"]).issubset(set(ct.song_ids))
