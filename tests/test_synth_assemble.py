import numpy as np
import pandas as pd

from synth.assemble import assemble_session
from synth.candidates import CandidateTable
from synth.engagement import EngagementModel
from synth.personas import Persona, SessionSpec

_SCHEMA_KEYS = {"user_id","session_id","ts_unix_ms","track_id","track_row","played_seconds",
                "track_duration_s","completed","skipped","liked","context_track_ids",
                "hour_of_day","day_of_week","is_weekend","session_pos"}


def _engagement():
    ev = pd.DataFrame({"completed": [1,0,1,0]*50, "skipped": [0,1,0,1]*50,
                       "playedMs": [200000,20000,200000,20000]*50, "trackDurationMs": [210000]*200,
                       "finalizeReason": ["TRACK_ENDED","USER_SKIPPED","TRACK_ENDED","SESSION_END"]*50})
    return EngagementModel.from_events(ev)


def _candidates():
    meta = pd.DataFrame({"song_id": ["uas-0","uas-1","uas-2"], "title":["A","B","C"],
                         "artist":["x"]*3, "album":[None]*3, "genre":["Pop"]*3, "year":[2000]*3,
                         "bpm":[120.0]*3, "energy":[0.5]*3})
    return CandidateTable(list(meta["song_id"]), {s:i for i,s in enumerate(meta["song_id"])},
                          np.zeros((3,960),dtype=np.float32), meta)


def test_assemble_produces_schema_rows():
    ct, eng = _candidates(), _engagement()
    spec = SessionSpec(Persona("explorer",0.6,0.3,0.9), "discovery", 3)
    rows = assemble_session(["uas-0","uas-1","uas-2"], spec, ct, eng,
                            user_id="u1", session_id="s1", start_ts_ms=1_700_000_000_000,
                            rng=np.random.default_rng(0))
    assert len(rows) == 3
    assert set(rows[0]) == _SCHEMA_KEYS
    assert [r["track_row"] for r in rows] == [0,1,2]
    assert [r["session_pos"] for r in rows] == [0,1,2]
    assert rows[0]["context_track_ids"] == []            # first has no context
    assert rows[2]["context_track_ids"] == ["uas-0","uas-1"]
    assert all(r["liked"] == 0 for r in rows)
    assert all(r["completed"] in (0,1) and r["completed"] != r["skipped"] for r in rows)
    assert all(0.0 <= r["played_seconds"] <= r["track_duration_s"] for r in rows)
