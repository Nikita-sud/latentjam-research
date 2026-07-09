import numpy as np
import pandas as pd

from synth.candidates import CandidateTable
from synth.personas import Persona, SessionSpec
from synth.reweight import candidate_subset, reweight_sessions


def _ct(n=200):
    meta = pd.DataFrame({"song_id":[f"uas-{i}" for i in range(n)], "title":[f"T{i}" for i in range(n)],
                         "artist":["a"]*n, "album":[None]*n, "genre":(["Pop"]*(n//2)+["Anime OST"]*(n-n//2)),
                         "year":[2000]*n, "bpm":[120.0]*n, "energy":[0.5]*n})
    return CandidateTable(list(meta["song_id"]), {s:i for i,s in enumerate(meta["song_id"])},
                          np.zeros((n,960),dtype=np.float32), meta)


def test_subset_size_and_variety():
    ct = _ct()
    spec = SessionSpec(Persona("explorer",0.6,0.3,0.9), "discovery", 12)
    a = candidate_subset(ct, spec, k=60, rng=np.random.default_rng(0))
    b = candidate_subset(ct, spec, k=60, rng=np.random.default_rng(1))
    assert len(a) == 60 and len(set(a)) == 60
    assert a != b  # different draws show different candidates (not a fixed head)


def test_reweight_flattens_head_and_bounds_drop():
    # 90% of events are one track -> heavy head
    ev = pd.DataFrame({"session_id": [f"s{i//5}" for i in range(500)],
                       "track_id": (["uas-0"]*450 + [f"uas-{i}" for i in range(50)])})
    out = reweight_sessions(ev, target_freq=None, rng=np.random.default_rng(0), max_drop=0.5)
    assert len(out) >= 0.5 * len(ev)  # never drops more than max_drop
    head_before = (ev["track_id"] == "uas-0").mean()
    head_after = (out["track_id"] == "uas-0").mean()
    assert head_after <= head_before  # head share not increased
