"""Assemble an ordered session into export-schema event rows with engagement labels."""

from __future__ import annotations

from datetime import UTC, datetime

from synth.candidates import CandidateTable
from synth.engagement import EngagementModel
from synth.personas import SessionSpec

_GAP_MS = 5_000  # small inter-track gap; play time dominates the timeline


def _time_feats(ts_ms: int) -> tuple[int, int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
    dow = dt.weekday()
    return dt.hour, dow, int(dow >= 5)


def assemble_session(
    song_ids: list[str], spec: SessionSpec, candidates: CandidateTable, engagement: EngagementModel,
    *, user_id: str, session_id: str, start_ts_ms: int, rng, track_duration_s: float = 210.0,
) -> list[dict]:
    labels = engagement.sample(len(song_ids), rng)
    rows: list[dict] = []
    ts = int(start_ts_ms)
    ctx: list[str] = []
    for pos, (sid, lbl) in enumerate(zip(song_ids, labels, strict=True)):
        hour, dow, wknd = _time_feats(ts)
        played_s = float(lbl.played_fraction * track_duration_s)
        rows.append({
            "user_id": user_id, "session_id": session_id, "ts_unix_ms": ts,
            "track_id": sid, "track_row": candidates.track_row[sid],
            "played_seconds": played_s, "track_duration_s": float(track_duration_s),
            "completed": int(lbl.completed), "skipped": int(lbl.skipped), "liked": 0,
            "context_track_ids": ctx[-4:],
            "hour_of_day": hour, "day_of_week": dow, "is_weekend": wknd, "session_pos": pos,
        })
        ctx.append(sid)
        ts += int(played_s * 1000) + _GAP_MS
    return rows
