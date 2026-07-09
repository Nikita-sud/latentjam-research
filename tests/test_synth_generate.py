import numpy as np
import pandas as pd

from synth.candidates import CandidateTable
from synth.generate import build_prompt, generate_session, parse_session
from synth.personas import Persona, SessionSpec


def _candidates(n=5):
    meta = pd.DataFrame({
        "song_id": [f"uas-{i}" for i in range(n)],
        "title": [f"T{i}" for i in range(n)], "artist": ["A"] * n,
        "album": [None] * n, "genre": ["Pop"] * n, "year": [2000] * n,
        "bpm": [120.0] * n, "energy": [0.5] * n,
    })
    return CandidateTable(list(meta["song_id"]), {s: i for i, s in enumerate(meta["song_id"])},
                          np.zeros((n, 960), dtype=np.float32), meta)


def test_parse_drops_invalid_and_maps():
    idx_map = {0: "uas-0", 1: "uas-1", 2: "uas-2"}
    out = parse_session({"track_indices": [1, 1, 9, 0]}, idx_map, session_len=10)
    assert out == ["uas-1", "uas-0"]  # dedup, drop out-of-range(9), keep order


def test_parse_truncates_to_session_len():
    idx_map = {0: "uas-0", 1: "uas-1", 2: "uas-2", 3: "uas-3"}
    out = parse_session({"track_indices": [0, 1, 2, 3]}, idx_map, session_len=2)
    assert out == ["uas-0", "uas-1"]  # more valid indices than session_len -> truncated


def test_generate_session_calls_llm_and_maps(monkeypatch):
    ct = _candidates(5)
    spec = SessionSpec(Persona("explorer", 0.6, 0.3, 0.9), "discovery", session_len=3)

    def fake_post(url, json, timeout):  # mimic Ollama /api/chat response
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"message": {"content": '{"track_indices": [2, 0, 4]}'}}
        return R()

    out = generate_session(spec, ct, model="qwen3.6:35b", subset=[0, 1, 2, 3, 4], http_post=fake_post, seed=0)
    assert out == ["uas-2", "uas-0", "uas-4"]


def test_generate_session_remaps_strict_subset_indices():
    # A strict, out-of-order subset: subset-local index 0 -> global row 4,
    # subset-local index 1 -> global row 1. The LLM sees a 0..len(subset)-1
    # renumbering and its indices must map back to the correct global song_ids.
    ct = _candidates(6)
    spec = SessionSpec(Persona("explorer", 0.6, 0.3, 0.9), "discovery", session_len=2)

    def fake_post(url, json, timeout):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {"message": {"content": '{"track_indices": [1, 0]}'}}
        return R()

    out = generate_session(spec, ct, model="m", subset=[4, 1], http_post=fake_post, seed=0)
    # subset-local 1 -> global row 1, subset-local 0 -> global row 4
    assert out == [ct.song_ids[1], ct.song_ids[4]]
    assert out == ["uas-1", "uas-4"]


def test_build_prompt_numbers_candidates():
    ct = _candidates(3)
    spec = SessionSpec(Persona("nostalgic", 0.5, 0.7, 0.35), "russian_throwback", 5)
    text, idx_map = build_prompt(spec, ct.meta.iloc[[0, 1, 2]])
    assert idx_map == {0: "uas-0", 1: "uas-1", 2: "uas-2"}
    assert "russian_throwback" in text and "0 |" in text and "T0" in text
