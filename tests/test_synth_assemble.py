from datetime import UTC, datetime

import numpy as np
import pandas as pd

from synth.assemble import assemble_session
from synth.candidates import CandidateTable
from synth.engagement import EngagementModel
from synth.personas import Persona, SessionSpec

_SCHEMA_KEYS = {"user_id","session_id","ts_unix_ms","track_id","track_row","played_seconds",
                "track_duration_s","completed","skipped","liked","context_track_ids",
                "hour_of_day","day_of_week","is_weekend","session_pos"}

# 2023-11-14T22:13:20Z (Tuesday -> is_weekend 0); +4 days = 2023-11-18 (Saturday -> is_weekend 1).
_WEEKDAY_TS = 1_700_000_000_000
_WEEKEND_TS = 1_700_345_600_000


def _engagement():
    ev = pd.DataFrame({"completed": [1,0,1,0]*50, "skipped": [0,1,0,1]*50,
                       "playedMs": [200000,20000,200000,20000]*50, "trackDurationMs": [210000]*200,
                       "finalizeReason": ["TRACK_ENDED","USER_SKIPPED","TRACK_ENDED","SESSION_END"]*50})
    return EngagementModel.from_events(ev)


def _candidates(n=3):
    ids = [f"uas-{i}" for i in range(n)]
    meta = pd.DataFrame({"song_id": ids, "title": ids, "artist":["x"]*n, "album":[None]*n,
                         "genre":["Pop"]*n, "year":[2000]*n, "bpm":[120.0]*n, "energy":[0.5]*n})
    return CandidateTable(list(meta["song_id"]), {s:i for i,s in enumerate(meta["song_id"])},
                          np.zeros((n,960),dtype=np.float32), meta)


def _spec(session_len):
    return SessionSpec(Persona("explorer",0.6,0.3,0.9), "discovery", session_len)


def test_assemble_produces_schema_rows():
    ct, eng = _candidates(), _engagement()
    rows = assemble_session(["uas-0","uas-1","uas-2"], _spec(3), ct, eng,
                            user_id="u1", session_id="s1", start_ts_ms=_WEEKDAY_TS,
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


def test_time_features_match_utc_derived_values_weekday_and_weekend():
    ct, eng = _candidates(), _engagement()
    seen_weekend = set()
    for start_ts_ms, expected_is_weekend in ((_WEEKDAY_TS, 0), (_WEEKEND_TS, 1)):
        rows = assemble_session(["uas-0","uas-1","uas-2"], _spec(3), ct, eng,
                                user_id="u1", session_id="s1", start_ts_ms=start_ts_ms,
                                rng=np.random.default_rng(0))
        dt = datetime.fromtimestamp(start_ts_ms / 1000, tz=UTC)
        assert rows[0]["hour_of_day"] == dt.hour
        assert rows[0]["day_of_week"] == dt.weekday()
        assert rows[0]["is_weekend"] == int(dt.weekday() >= 5) == expected_is_weekend
        seen_weekend.add(rows[0]["is_weekend"])
    assert seen_weekend == {0, 1}  # both branches exercised


def test_timestamp_advances_across_events():
    ct, eng = _candidates(8), _engagement()
    ids = [f"uas-{i}" for i in range(8)]
    rows = assemble_session(ids, _spec(8), ct, eng,
                            user_id="u1", session_id="s1", start_ts_ms=_WEEKDAY_TS,
                            rng=np.random.default_rng(3))
    ts = [r["ts_unix_ms"] for r in rows]
    assert ts[0] == _WEEKDAY_TS
    assert all(ts[i + 1] >= ts[i] for i in range(len(ts) - 1))  # non-decreasing
    # A fixed inter-track gap is always added on top of play time, so ts strictly increases
    # every step; a regression that stops advancing ts (all-equal timestamps) fails here.
    assert all(ts[i + 1] > ts[i] for i in range(len(ts) - 1))


def test_context_window_truncates_to_last_four():
    ct, eng = _candidates(8), _engagement()
    ids = [f"uas-{i}" for i in range(8)]
    rows = assemble_session(ids, _spec(8), ct, eng,
                            user_id="u1", session_id="s1", start_ts_ms=_WEEKDAY_TS,
                            rng=np.random.default_rng(1))
    assert len(rows) == 8
    # From the 5th event (pos=4) onward the window is exactly the last 4 predecessors, in order.
    for pos in range(4, len(rows)):
        assert len(rows[pos]["context_track_ids"]) == 4
        assert rows[pos]["context_track_ids"] == ids[pos - 4:pos]
    # The load-bearing case: pos=5 -> song_ids[1:5], NOT the full prefix.
    assert rows[5]["context_track_ids"] == ids[1:5]


def test_passthrough_fields_and_nondefault_duration():
    ct, eng = _candidates(), _engagement()
    rows = assemble_session(["uas-0","uas-1","uas-2"], _spec(3), ct, eng,
                            user_id="user-42", session_id="sess-99", start_ts_ms=_WEEKDAY_TS,
                            rng=np.random.default_rng(0), track_duration_s=90.0)
    assert all(r["user_id"] == "user-42" for r in rows)      # not swapped with session_id
    assert all(r["session_id"] == "sess-99" for r in rows)
    assert all(r["track_duration_s"] == 90.0 for r in rows)
    assert all(0.0 <= r["played_seconds"] <= 90.0 for r in rows)
