"""LLM session generator — ordered candidate indices via Ollama guided-JSON."""

from __future__ import annotations

import json

import pandas as pd
import requests

from synth.candidates import CandidateTable
from synth.personas import SessionSpec

OLLAMA_URL = "http://localhost:11434/api/chat"

SESSION_SCHEMA = {
    "type": "object",
    "properties": {"track_indices": {"type": "array", "items": {"type": "integer"}}},
    "required": ["track_indices"],
}

_GOAL_HINT = {
    "workout": "a high-energy workout set that builds and sustains intensity",
    "focus": "a long, low-distraction focus/study set with steady mood",
    "evening_chill": "a mellow evening wind-down that gradually calms",
    "anime_binge": "an anime/game OST listening run with tonal continuity",
    "russian_throwback": "a Russian-language pop/rock throwback session",
    "disco_eurodance_party": "an upbeat disco/eurodance party set",
    "film_score_ambient": "an ambient film-score background set",
    "discovery": "an exploratory set that ventures into the long tail",
}


def build_prompt(spec: SessionSpec, candidates_meta: pd.DataFrame) -> tuple[str, dict]:
    index_to_song: dict[int, str] = {}
    lines = []
    for i, (_, row) in enumerate(candidates_meta.reset_index(drop=True).iterrows()):
        index_to_song[i] = row["song_id"]
        bits = [
            str(row.get("genre") or "?"),
            str(int(row["year"])) if pd.notna(row.get("year")) else "?",
        ]
        if pd.notna(row.get("bpm")):
            bits.append(f"{int(row['bpm'])}bpm")
        lines.append(f"{i} | {row['title']} — {row['artist']} [{', '.join(bits)}]")
    p = spec.persona
    prompt = (
        "You are simulating a real listener choosing what to play next from THEIR OWN library.\n"
        f"Persona: {p.name} (activity={p.activity:.2f}, conformity={p.conformity:.2f}, "
        f"diversity={p.diversity:.2f}).\n"
        f"Intent ({spec.goal}): {_GOAL_HINT[spec.goal]}.\n"
        f"Build a coherent listening session of about {spec.session_len} tracks by choosing from the "
        "numbered candidates below. Order them the way a real person would actually play them "
        "(mood/energy arc, artist/genre coherence, occasional variety per the diversity dial). "
        "Return ONLY the chosen indices in play order.\n\n"
        "Candidates:\n" + "\n".join(lines)
    )
    return prompt, index_to_song


def parse_session(response_json: dict, index_to_song: dict[int, str], session_len: int) -> list[str]:
    idxs = response_json.get("track_indices", [])
    out, seen = [], set()
    for i in idxs:
        if not isinstance(i, int) or i not in index_to_song or i in seen:
            continue
        seen.add(i)
        out.append(index_to_song[i])
        if len(out) >= session_len:
            break
    return out


def generate_session(
    spec: SessionSpec,
    candidates: CandidateTable,
    *,
    model: str,
    subset: list[int],
    http_post=requests.post,
    seed: int = 0,
) -> list[str]:
    meta = candidates.meta.iloc[subset]
    prompt, index_to_song = build_prompt(spec, meta)
    resp = http_post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "format": SESSION_SCHEMA,
            "stream": False,
            "options": {"seed": seed, "temperature": 0.8},
        },
        timeout=300,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return parse_session(json.loads(content), index_to_song, spec.session_len)
