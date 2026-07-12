# Synthetic Listening — Generation & Export Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Plan 1 data foundation into a validated synthetic listening corpus — a local LLM generates culturally-coherent track sequences over the library, engagement labels are derived and calibrated, popularity bias is corrected, the corpus passes the validation gate, and it exports as `synth_listening_v3.parquet` in the exact schema `train_history.py` consumes.

**Architecture:** Reuse the existing `synthesize_listening.py` output schema and event scaffolding, but replace its heuristic next-track sampler with an **LLM-driven session generator** (Ollama guided-JSON over manifest metadata + persona/goal conditioning) and its heuristic skip logic with Plan 1's `EngagementModel`. Embeddings + `track_row` come from the **960-d `TrackEmbeddingEntity`** (the app's shipping space), aligned to the Plan 1 manifest. New code in `src/synth/`; a driver CLI in `scripts/`.

**Tech Stack:** Python 3.12, pandas/numpy/pyarrow, `requests` (Ollama HTTP API — already a dep), Ollama running locally with `qwen3.6:35b` (+ `aya-expanse:32b` for rus/anime), pytest. Consumes Plan 1: `synth.cachedfile`, `synth.manifest`, `synth.engagement.EngagementModel`, `synth.validate.validate_corpus`.

## Global Constraints

- src-layout: importable code in `src/synth/`, tests `tests/test_synth_*.py`, CLIs in `scripts/`. `python -m pytest` from repo root; ruff `line-length=100`.
- **Export schema is FIXED** — `synth_listening_v3.parquet` must have exactly these columns/types (match `synth_listening_v1/v2.parquet`, consumed by `src/predictor/train_history.py`): `user_id:str, session_id:str, ts_unix_ms:int64, track_id:str, track_row:int64, played_seconds:float64, track_duration_s:float64, completed:int64, skipped:int64, liked:int64, context_track_ids:list<str>, hour_of_day:int64, day_of_week:int64, is_weekend:int64, session_pos:int64`. `track_id` is the `song_id` (Music.UID). `track_row` indexes the 960-d embedding matrix. `liked` is always `0` (favorites feature removed).
- **Embeddings/track_row come from the DB** `TrackEmbeddingEntity` (960-d, `db-backups/playback_persistence-2026-07-09.db`), aligned to the Plan 1 manifest `song_id`. Deduplicate the known 8/814 duplicate `song_id` rows.
- **The LLM emits track ORDERING only** (a list of candidate indices) — never engagement labels. Labels come from `EngagementModel` (Plan 1), calibrated to the real 3,194 events.
- **Every corpus must pass `validate_corpus` (Plan 1 gate) before export.** No gate pass → no parquet.
- LLM calls hit the local Ollama HTTP API (`http://localhost:11434/api/chat`) with `format` = a JSON schema. Unit tests MUST mock the LLM (no live model in tests); LLM quality is validated by the pilot + the gate, not by unit tests.
- **Commit ONLY each task's own files** (explicit `git add` paths). The repo has unrelated uncommitted changes — never `git add -A`/`.`/`-a`.

---

### Task 1: Candidate table — 960-d embeddings + track_row aligned to the manifest (`src/synth/candidates.py`)

Build the single source of truth the generator and exporter share: for each library track, its `song_id`, `track_row` (matrix index), 960-d L2-normed embedding, and display metadata.

**Files:** Create `src/synth/candidates.py`; Test `tests/test_synth_candidates.py`.

**Interfaces:**
- Consumes: `synth.manifest.build_manifest` (Plan 1), the DB `TrackEmbeddingEntity`.
- Produces:
  - `@dataclass CandidateTable(song_ids: list[str], track_row: dict[str,int], matrix: numpy.ndarray, meta: pandas.DataFrame)` — `matrix` is `(N, 960)` float32 L2-normalized; `meta` has one row per candidate with `["song_id","title","artist","album","genre","year","bpm","energy"]`, indexed 0..N-1 matching `matrix`/`track_row`.
  - `def build_candidate_table(manifest: pandas.DataFrame, playback_db: str) -> CandidateTable` — joins manifest to embeddings by `song_id`, drops tracks with no embedding, **deduplicates `song_id`** (keep first), assigns `track_row = 0..N-1`.
  - `def load_embeddings(playback_db: str) -> dict[str, numpy.ndarray]` — decode `TrackEmbeddingEntity.embedding` (little-endian float32, 960 dims) keyed by `songUid`.

- [ ] **Step 1: Write the failing test**

`tests/test_synth_candidates.py`:
```python
import sqlite3
import struct

import numpy as np
import pandas as pd

from synth.candidates import build_candidate_table, load_embeddings


def _emb_blob(vec):
    return struct.pack("<%df" % len(vec), *vec)


def _make_playback(path, rows):  # rows: list[(song_id, vec)]
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE TrackEmbeddingEntity (songUid TEXT PRIMARY KEY, modelVersion TEXT, "
                "embedding BLOB, embeddedAtMs INTEGER, tempo REAL, energy REAL)")
    for sid, vec in rows:
        con.execute("INSERT INTO TrackEmbeddingEntity VALUES (?,?,?,?,?,?)",
                    (sid, "v3", _emb_blob(vec), 0, 90.0, 0.5))
    con.commit(); con.close()


def test_load_embeddings_roundtrip(tmp_path):
    db = tmp_path / "p.db"
    vec = list(np.arange(960, dtype=np.float32) / 960.0)
    _make_playback(db, [("uas-a", vec)])
    emb = load_embeddings(str(db))
    assert emb["uas-a"].shape == (960,)
    assert np.allclose(emb["uas-a"], vec, atol=1e-6)


def test_build_candidate_table_aligns_and_dedupes(tmp_path):
    db = tmp_path / "p.db"
    v = lambda k: list((np.arange(960, dtype=np.float32) + k) / 960.0)
    _make_playback(db, [("uas-a", v(0)), ("uas-b", v(1))])
    manifest = pd.DataFrame({
        "song_id": ["uas-a", "uas-b", "uas-b", "uas-z"],  # dup uas-b; uas-z has no embedding
        "title": ["A", "B", "B", "Z"], "artist": ["x", "y", "y", "z"],
        "album": [None]*4, "genre": ["Pop", "Rock", "Rock", "Jazz"],
        "year": [2000]*4, "language": [None]*4, "bpm": [90.0]*4, "energy": [0.5]*4,
    })
    ct = build_candidate_table(manifest, str(db))
    assert ct.song_ids == ["uas-a", "uas-b"]                 # dedup + drop no-embedding
    assert ct.matrix.shape == (2, 960)
    assert np.allclose(np.linalg.norm(ct.matrix, axis=1), 1.0, atol=1e-5)  # L2-normed
    assert ct.track_row == {"uas-a": 0, "uas-b": 1}
    assert list(ct.meta["song_id"]) == ["uas-a", "uas-b"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_synth_candidates.py -v` — Expected: FAIL, `ModuleNotFoundError: No module named 'synth.candidates'`.

- [ ] **Step 3: Implement `src/synth/candidates.py`**

```python
"""Candidate table: 960-d embeddings + track_row aligned to the Plan 1 manifest."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

_EMBED_DIM = 960
_META_COLS = ["song_id", "title", "artist", "album", "genre", "year", "bpm", "energy"]


def load_embeddings(playback_db: str) -> dict[str, np.ndarray]:
    con = sqlite3.connect(f"file:{playback_db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT songUid, embedding FROM TrackEmbeddingEntity").fetchall()
    finally:
        con.close()
    out: dict[str, np.ndarray] = {}
    for song_id, blob in rows:
        vec = np.frombuffer(blob, dtype="<f4", count=_EMBED_DIM).astype(np.float32)
        out[song_id] = vec
    return out


@dataclass
class CandidateTable:
    song_ids: list[str]
    track_row: dict[str, int]
    matrix: np.ndarray
    meta: pd.DataFrame


def build_candidate_table(manifest: pd.DataFrame, playback_db: str) -> CandidateTable:
    emb = load_embeddings(playback_db)
    seen: set[str] = set()
    kept_rows, vectors = [], []
    for r in manifest.itertuples(index=False):
        sid = r.song_id
        if sid in seen or sid not in emb:
            continue
        seen.add(sid)
        kept_rows.append(r)
        vectors.append(emb[sid])
    matrix = np.vstack(vectors).astype(np.float32)
    matrix /= (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    meta = pd.DataFrame([{c: getattr(r, c) for c in _META_COLS} for r in kept_rows])
    song_ids = list(meta["song_id"])
    return CandidateTable(
        song_ids=song_ids,
        track_row={sid: i for i, sid in enumerate(song_ids)},
        matrix=matrix,
        meta=meta.reset_index(drop=True),
    )
```

- [ ] **Step 4: Run to verify it passes** — Run: `python -m pytest tests/test_synth_candidates.py -v` — Expected: PASS.

- [ ] **Step 5: Smoke against real data**

Run:
```bash
python -c "from synth.manifest import build_manifest; from synth.candidates import build_candidate_table; \
m=build_manifest('$SCRATCH/phone-sync-2026-07-09/music_cache.db','../db-backups/playback_persistence-2026-07-09.db'); \
ct=build_candidate_table(m,'../db-backups/playback_persistence-2026-07-09.db'); \
print('candidates', len(ct.song_ids), 'matrix', ct.matrix.shape)"
```
Expected: ~810 candidates, matrix `(~810, 960)` (deduped from 814). Confirms alignment.

- [ ] **Step 6: Commit**
```bash
ruff check src/synth/candidates.py tests/test_synth_candidates.py
git add src/synth/candidates.py tests/test_synth_candidates.py
git commit -m "feat(synth): candidate table (960-d embeddings + track_row from DB)"
```

---

### Task 2: Persona × goal grid (`src/synth/personas.py`)

Pre-structure diversity so it doesn't rely on sampling temperature (spec §5 stage 3).

**Files:** Create `src/synth/personas.py`; Test `tests/test_synth_personas.py`.

**Interfaces:**
- Produces:
  - `@dataclass Persona(name: str, activity: float, conformity: float, diversity: float)` (dials in [0,1]).
  - `@dataclass SessionSpec(persona: Persona, goal: str, session_len: int)`.
  - `GOALS: list[str]` — library-tailored: `["workout","focus","evening_chill","anime_binge","russian_throwback","disco_eurodance_party","film_score_ambient","discovery"]`.
  - `def build_grid(n_sessions: int, rng: numpy.random.Generator, goal_weights: dict[str,float] | None = None) -> list[SessionSpec]` — samples `n_sessions` specs from a persona archetype set × weighted goals; session_len drawn per-goal (e.g. workout shorter, focus longer). Default `goal_weights` over-samples the long-tail (`anime_binge`, `russian_throwback`, `discovery`) to counter head bias.

- [ ] **Step 1: Write the failing test**

`tests/test_synth_personas.py`:
```python
import numpy as np

from synth.personas import GOALS, SessionSpec, build_grid


def test_grid_size_and_shape():
    grid = build_grid(500, np.random.default_rng(0))
    assert len(grid) == 500
    assert all(isinstance(s, SessionSpec) for s in grid)
    assert all(s.goal in GOALS for s in grid)
    assert all(2 <= s.session_len <= 60 for s in grid)
    assert all(0.0 <= s.persona.diversity <= 1.0 for s in grid)


def test_grid_is_deterministic_per_seed():
    a = build_grid(50, np.random.default_rng(7))
    b = build_grid(50, np.random.default_rng(7))
    assert [(s.goal, s.session_len, s.persona.name) for s in a] == \
           [(s.goal, s.session_len, s.persona.name) for s in b]


def test_goal_weights_bias_the_mix():
    grid = build_grid(2000, np.random.default_rng(1), goal_weights={"discovery": 10.0})
    frac = sum(s.goal == "discovery" for s in grid) / len(grid)
    assert frac > 0.4  # heavily up-weighted goal dominates
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/test_synth_personas.py -v` — Expected FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/synth/personas.py`**

```python
"""Persona x goal grid — pre-structured diversity for synthetic sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GOALS = [
    "workout", "focus", "evening_chill", "anime_binge",
    "russian_throwback", "disco_eurodance_party", "film_score_ambient", "discovery",
]

# Session length range per goal (min, max).
_GOAL_LEN = {
    "workout": (8, 20), "focus": (15, 45), "evening_chill": (6, 18),
    "anime_binge": (10, 30), "russian_throwback": (8, 25),
    "disco_eurodance_party": (10, 30), "film_score_ambient": (6, 20), "discovery": (8, 25),
}

# Persona archetypes (activity, conformity, diversity dials).
_ARCHETYPES = [
    ("mainstream_regular", 0.7, 0.8, 0.3),
    ("explorer", 0.6, 0.3, 0.9),
    ("focused_listener", 0.4, 0.6, 0.4),
    ("nostalgic", 0.5, 0.7, 0.35),
    ("power_user", 0.9, 0.5, 0.7),
]

# Default goal weights: over-sample the long-tail to counter head bias.
_DEFAULT_GOAL_WEIGHTS = {
    "workout": 1.0, "focus": 1.2, "evening_chill": 1.0, "anime_binge": 1.5,
    "russian_throwback": 1.5, "disco_eurodance_party": 1.0,
    "film_score_ambient": 0.8, "discovery": 1.5,
}


@dataclass
class Persona:
    name: str
    activity: float
    conformity: float
    diversity: float


@dataclass
class SessionSpec:
    persona: Persona
    goal: str
    session_len: int


def build_grid(n_sessions: int, rng: np.random.Generator, goal_weights: dict[str, float] | None = None) -> list[SessionSpec]:
    weights = dict(_DEFAULT_GOAL_WEIGHTS)
    if goal_weights:
        weights.update(goal_weights)
    goals = np.array(GOALS)
    probs = np.array([weights[g] for g in GOALS], dtype=float)
    probs /= probs.sum()
    specs: list[SessionSpec] = []
    for _ in range(n_sessions):
        arc = _ARCHETYPES[rng.integers(0, len(_ARCHETYPES))]
        # small per-session jitter on the dials
        persona = Persona(
            name=arc[0],
            activity=float(np.clip(arc[1] + rng.normal(0, 0.05), 0, 1)),
            conformity=float(np.clip(arc[2] + rng.normal(0, 0.05), 0, 1)),
            diversity=float(np.clip(arc[3] + rng.normal(0, 0.05), 0, 1)),
        )
        goal = str(rng.choice(goals, p=probs))
        lo, hi = _GOAL_LEN[goal]
        specs.append(SessionSpec(persona=persona, goal=goal, session_len=int(rng.integers(lo, hi + 1))))
    return specs
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.
- [ ] **Step 5: Commit**
```bash
ruff check src/synth/personas.py tests/test_synth_personas.py
git add src/synth/personas.py tests/test_synth_personas.py
git commit -m "feat(synth): persona x goal grid"
```

---

### Task 3: LLM session generator (`src/synth/generate.py`)

Given a `SessionSpec` and a candidate subset, ask the local LLM (Ollama, guided-JSON) for an ordered list of candidate **indices**; validate and map to `song_id`/`track_row`. The prompt is iterated in the pilot (Task 6); this task's tests are hermetic (mock the HTTP call) and verify parsing/validation/mapping only.

**Files:** Create `src/synth/generate.py`; Test `tests/test_synth_generate.py`.

**Interfaces:**
- Consumes: `synth.candidates.CandidateTable`, `synth.personas.SessionSpec`.
- Produces:
  - `def build_prompt(spec, candidates_meta: pandas.DataFrame) -> tuple[str, dict]` — returns (system+user prompt text, the numbered candidate map `{index: song_id}`). Candidates are presented as a numbered list `idx | title — artist [genre, year, bpm]`.
  - `SESSION_SCHEMA: dict` — JSON schema: `{"type":"object","properties":{"track_indices":{"type":"array","items":{"type":"integer"}}},"required":["track_indices"]}`.
  - `def parse_session(response_json: dict, index_to_song: dict[int,str], session_len: int) -> list[str]` — extract `track_indices`, drop out-of-range/duplicate indices, map to `song_id`s, truncate/allow short to `session_len`. Returns ordered `song_id`s.
  - `def generate_session(spec, candidates: CandidateTable, *, model: str, subset: list[int], http_post=..., seed: int) -> list[str]` — builds the prompt over `subset` candidate rows, calls Ollama `/api/chat` with `format=SESSION_SCHEMA`, parses. `http_post` is injectable for tests.

- [ ] **Step 1: Write the failing test (mock the LLM)**

`tests/test_synth_generate.py`:
```python
import numpy as np
import pandas as pd

from synth.generate import build_prompt, generate_session, parse_session
from synth.personas import SessionSpec, Persona
from synth.candidates import CandidateTable


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


def test_generate_session_calls_llm_and_maps(monkeypatch):
    ct = _candidates(5)
    spec = SessionSpec(Persona("explorer", 0.6, 0.3, 0.9), "discovery", session_len=3)

    def fake_post(url, json, timeout):  # mimic Ollama /api/chat response
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self_):
                return {"message": {"content": '{"track_indices": [2, 0, 4]}'}}
        return R()

    out = generate_session(spec, ct, model="qwen3.6:35b", subset=[0, 1, 2, 3, 4], http_post=fake_post, seed=0)
    assert out == ["uas-2", "uas-0", "uas-4"]


def test_build_prompt_numbers_candidates():
    ct = _candidates(3)
    spec = SessionSpec(Persona("nostalgic", 0.5, 0.7, 0.35), "russian_throwback", 5)
    text, idx_map = build_prompt(spec, ct.meta.iloc[[0, 1, 2]])
    assert idx_map == {0: "uas-0", 1: "uas-1", 2: "uas-2"}
    assert "russian_throwback" in text and "0 |" in text and "T0" in text
```

- [ ] **Step 2: Run to verify it fails** — Expected FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `src/synth/generate.py`**

```python
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
        bits = [str(row.get("genre") or "?"), str(int(row["year"])) if pd.notna(row.get("year")) else "?"]
        if pd.notna(row.get("bpm")):
            bits.append(f"{int(row['bpm'])}bpm")
        lines.append(f"{i} | {row['title']} — {row['artist']} [{', '.join(bits)}]")
    p = spec.persona
    prompt = (
        "You are simulating a real listener choosing what to play next from THEIR OWN library.\n"
        f"Persona: {p.name} (activity={p.activity:.2f}, conformity={p.conformity:.2f}, "
        f"diversity={p.diversity:.2f}).\n"
        f"Intent: {_GOAL_HINT[spec.goal]}.\n"
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
    spec: SessionSpec, candidates: CandidateTable, *, model: str, subset: list[int],
    http_post=requests.post, seed: int = 0,
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
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (LLM mocked).

- [ ] **Step 5: Live spike (manual, not a committed test) — verify Ollama guided-JSON actually returns valid indices**

With Ollama running (`ollama serve`), run a one-off:
```bash
python -c "
from synth.manifest import build_manifest; from synth.candidates import build_candidate_table
from synth.personas import build_grid; from synth.generate import generate_session
import numpy as np
m=build_manifest('$SCRATCH/phone-sync-2026-07-09/music_cache.db','../db-backups/playback_persistence-2026-07-09.db')
ct=build_candidate_table(m,'../db-backups/playback_persistence-2026-07-09.db')
spec=build_grid(1, np.random.default_rng(0))[0]
subset=list(range(min(120,len(ct.song_ids))))
print(spec.goal, '->', generate_session(spec, ct, model='qwen3.6:35b', subset=subset, seed=0)[:10])"
```
Expected: a short ordered list of real `song_id`s. If Ollama rejects the `format` schema or returns non-integer indices, adjust the prompt/schema here before proceeding (this is the prompt-iteration point). Report what the model returned.

- [ ] **Step 6: Commit**
```bash
ruff check src/synth/generate.py tests/test_synth_generate.py
git add src/synth/generate.py tests/test_synth_generate.py
git commit -m "feat(synth): LLM session generator (Ollama guided-JSON, index-based)"
```

---

### Task 4: Event assembly (`src/synth/assemble.py`)

Turn an ordered session (`song_id`s) + its `SessionSpec` + a start timestamp into rows in the fixed export schema, applying `EngagementModel` for labels.

**Files:** Create `src/synth/assemble.py`; Test `tests/test_synth_assemble.py`.

**Interfaces:**
- Consumes: `synth.engagement.EngagementModel` (Plan 1), `synth.candidates.CandidateTable`, `synth.personas.SessionSpec`.
- Produces:
  - `def assemble_session(song_ids: list[str], spec, candidates: CandidateTable, engagement: EngagementModel, *, user_id: str, session_id: str, start_ts_ms: int, rng, track_duration_s: float = 210.0) -> list[dict]` — one dict per event with EXACTLY the export-schema keys. `track_row` from `candidates.track_row`; `context_track_ids` = the last up-to-4 predecessor `track_id`s in the session; `played_seconds = played_fraction * track_duration_s`; `completed`/`skipped` from `EngagementModel.sample(len)`; `hour_of_day/day_of_week/is_weekend` from `start_ts_ms` advanced per track; `liked=0`.

- [ ] **Step 1: Write the failing test**

`tests/test_synth_assemble.py`:
```python
import numpy as np
import pandas as pd

from synth.assemble import assemble_session
from synth.engagement import EngagementModel
from synth.personas import SessionSpec, Persona
from synth.candidates import CandidateTable

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
```

- [ ] **Step 2: Run to verify it fails** — Expected FAIL.

- [ ] **Step 3: Implement `src/synth/assemble.py`**

```python
"""Assemble an ordered session into export-schema event rows with engagement labels."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from synth.candidates import CandidateTable
from synth.engagement import EngagementModel
from synth.personas import SessionSpec

_GAP_MS = 5_000  # small inter-track gap; play time dominates the timeline


def _time_feats(ts_ms: int) -> tuple[int, int, int]:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
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
    for pos, (sid, lbl) in enumerate(zip(song_ids, labels)):
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
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.
- [ ] **Step 5: Commit**
```bash
ruff check src/synth/assemble.py tests/test_synth_assemble.py
git add src/synth/assemble.py tests/test_synth_assemble.py
git commit -m "feat(synth): assemble sessions into export-schema events"
```

---

### Task 5: Popularity-bias reweighting (`src/synth/reweight.py`)

Correct head bias: subset selection per session should not always show the most popular tracks, and the final corpus is reweighted toward the real catalog demand (spec §5 stage 6).

**Files:** Create `src/synth/reweight.py`; Test `tests/test_synth_reweight.py`.

**Interfaces:**
- Produces:
  - `def candidate_subset(candidates: CandidateTable, spec: SessionSpec, *, k: int, rng) -> list[int]` — choose `k` candidate row indices to show the LLM for this session, biased by goal/genre relevance but sampling the long tail per the persona `diversity` dial (never always the same head tracks).
  - `def reweight_sessions(events: pandas.DataFrame, *, target_freq: dict[str,float] | None, rng, max_drop: float = 0.5) -> pandas.DataFrame` — post-hoc drop/keep whole sessions so per-track frequency moves toward `target_freq` (or flattens the head when `None`); never drops more than `max_drop` of sessions.

- [ ] **Step 1: Write the failing test**

`tests/test_synth_reweight.py`:
```python
import numpy as np
import pandas as pd

from synth.reweight import candidate_subset, reweight_sessions
from synth.personas import SessionSpec, Persona
from synth.candidates import CandidateTable


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
```

- [ ] **Step 2: Run to verify it fails** — Expected FAIL.

- [ ] **Step 3: Implement `src/synth/reweight.py`** (candidate_subset: relevance-biased but diversity-sampled selection; reweight_sessions: down-sample sessions dominated by over-represented tracks, bounded by `max_drop`). Write the implementation to satisfy the tests: `candidate_subset` mixes a relevance-ranked slice (by genre/goal match) with a random long-tail slice sized by `spec.persona.diversity`; `reweight_sessions` computes per-track frequency, assigns each session a keep-probability inversely proportional to its tracks' over-representation vs `target_freq` (uniform head-flatten when `None`), samples sessions to keep, and stops dropping once `max_drop` is reached.

```python
"""Popularity-bias correction: diverse candidate subsets + post-hoc session reweighting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synth.candidates import CandidateTable
from synth.personas import SessionSpec

# Goal -> genres it favors (soft relevance prior; unmatched genres still reachable via the tail).
_GOAL_GENRES = {
    "anime_binge": {"Anime OST", "Game OST"}, "film_score_ambient": {"Film Score"},
    "russian_throwback": {"Russian Pop", "Russian Rock"},
    "disco_eurodance_party": {"Disco", "Eurodance", "Dance-pop"}, "workout": {"Hip-Hop", "Hard Rock", "Eurodance"},
}


def candidate_subset(candidates: CandidateTable, spec: SessionSpec, *, k: int, rng) -> list[int]:
    n = len(candidates.song_ids)
    k = min(k, n)
    fav = _GOAL_GENRES.get(spec.goal, set())
    genres = candidates.meta["genre"].fillna("")
    relevant = [i for i in range(n) if genres.iloc[i] in fav]
    rng.shuffle(relevant)
    n_relevant = int(round(k * (1.0 - spec.persona.diversity)))
    picks = list(relevant[:n_relevant])
    remaining = [i for i in range(n) if i not in set(picks)]
    rng.shuffle(remaining)
    picks += remaining[: k - len(picks)]
    return picks[:k]


def reweight_sessions(events: pd.DataFrame, *, target_freq, rng, max_drop: float = 0.5) -> pd.DataFrame:
    freq = events["track_id"].value_counts(normalize=True).to_dict()
    if target_freq is None:
        target_freq = {t: 1.0 / len(freq) for t in freq}  # flat head
    # per-session keep score: lower when its tracks are over-represented
    keep_prob: dict[str, float] = {}
    for sid, grp in events.groupby("session_id"):
        over = np.mean([freq.get(t, 0) / max(target_freq.get(t, 1e-9), 1e-9) for t in grp["track_id"]])
        keep_prob[sid] = float(np.clip(1.0 / max(over, 1e-6), 0.05, 1.0))
    session_ids = list(keep_prob)
    order = list(session_ids)
    rng.shuffle(order)
    min_keep = int(np.ceil((1.0 - max_drop) * len(events)))
    kept, kept_rows = set(), 0
    total = len(events)
    counts = events.groupby("session_id").size().to_dict()
    for sid in order:
        if rng.random() < keep_prob[sid] or (total - kept_rows) <= min_keep:
            kept.add(sid)
            kept_rows += counts[sid]
        else:
            total -= counts[sid]
    return events[events["session_id"].isin(kept)].reset_index(drop=True)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.
- [ ] **Step 5: Commit**
```bash
ruff check src/synth/reweight.py tests/test_synth_reweight.py
git add src/synth/reweight.py tests/test_synth_reweight.py
git commit -m "feat(synth): popularity-bias reweighting"
```

---

### Task 6: Corpus driver — generate, gate, export (`scripts/synth_generate_corpus.py`)

Wire everything: grid → per-session (subset → LLM order → assemble) → concat → reweight → `validate_corpus` gate → export `synth_listening_v3.parquet`. Includes a Mac pilot mode.

**Files:** Create `scripts/synth_generate_corpus.py`; Test `tests/test_synth_corpus_driver.py`.

**Interfaces:**
- Consumes all of Tasks 1–5 + `synth.engagement`, `synth.validate`, `synth.manifest`.
- Produces:
  - `def generate_corpus(candidates, engagement, grid, *, model, k, rng, generate_fn) -> pandas.DataFrame` — the pure pipeline (LLM call injected via `generate_fn` for tests): for each `SessionSpec`, `subset = candidate_subset(...)`, `song_ids = generate_fn(spec, candidates, subset)`, `rows = assemble_session(...)`; concat to a DataFrame. Skips empty sessions.
  - CLI `scripts/synth_generate_corpus.py`: `--music-cache`, `--playback-db`, `--n-sessions`, `--model`, `--audit-csv`, `--out` (default `models/predictor/synth_listening_v3.parquet`), `--seed`. Builds manifest+candidates+engagement(from the real DB), runs `generate_corpus` with the real Ollama `generate_session`, runs `reweight_sessions`, then `validate_corpus(sessions=<song_id+completed view>, manifest, real_events)` — **abort (non-zero exit, no parquet) if the gate fails**, printing the report; on pass, writes the parquet and prints coverage/KL/counts.

- [ ] **Step 1: Write the failing test (LLM injected)**

`tests/test_synth_corpus_driver.py`:
```python
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
driver = importlib.util.module_from_spec(spec); sys.modules["synth_generate_corpus"] = driver
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
```

- [ ] **Step 2: Run to verify it fails** — Expected FAIL.

- [ ] **Step 3: Implement `scripts/synth_generate_corpus.py`** — the pure `generate_corpus` (loop grid, per spec: subset → `generate_fn` → `assemble_session` with `user_id=persona.name+idx`, `session_id=f"synth-{i}"`, a rolling `start_ts_ms`) returning the concatenated DataFrame; plus the click CLI that builds real inputs, uses the real `generate_session` as `generate_fn`, applies `reweight_sessions`, runs the gate, and writes/aborts. (Full code: compose the Task 1–5 functions; `real_events = load_real_events(playback_db)` from `synth.engagement`; the gate's `sessions` arg is `df[["track_id","completed"]].rename(columns={"track_id":"song_id"})`.)

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Mac pilot run (the real integration test — Ollama required)**

With Ollama serving `qwen3.6:35b`:
```bash
python scripts/synth_generate_corpus.py \
  --music-cache "$SCRATCH/phone-sync-2026-07-09/music_cache.db" \
  --playback-db ../db-backups/playback_persistence-2026-07-09.db \
  --n-sessions 200 --model qwen3.6:35b --out data/synth/pilot_v3.parquet --seed 0
```
Expected: ~200 sessions generated, reweighted, and EITHER a printed gate PASS + parquet written, OR a gate FAIL with the KL/coverage/completion report (then iterate the Task 3 prompt / Task 5 weights and re-run — this is the pilot loop from spec §6). Report the gate metrics. Do NOT commit the parquet (it's under `data/synth/`, git-ignored by `/data/*`).

- [ ] **Step 6: Commit**
```bash
ruff check scripts/synth_generate_corpus.py tests/test_synth_corpus_driver.py
git add scripts/synth_generate_corpus.py tests/test_synth_corpus_driver.py
git commit -m "feat(synth): corpus driver — generate, reweight, gate, export v3"
```

---

## Self-Review

- **Spec coverage:** Task 1 ⇒ candidate/embedding source (960-d, spec §4 embeddings); Task 2 ⇒ stage 3 persona×goal grid; Task 3 ⇒ stage 4 LLM generation (guided-JSON, index selection = the songId-enum intent without a huge enum); Task 4 ⇒ stage 5 engagement application (reuses Plan 1's `EngagementModel`); Task 5 ⇒ stage 6 popularity reweighting; Task 6 ⇒ stage 6 gate + stage 7 export. The **downstream eval** (train on v3, hit-rate vs retrieval baseline on held-out real events — spec §9) is **Plan 3**, since it wires `train_history.py` + a metric harness, not this corpus-generation increment.
- **Placeholder scan:** Tasks 3 and 6 intentionally defer LLM-prompt/weight *tuning* to the pilot (Task 3 Step 5, Task 6 Step 5) — this is genuine iterate-against-the-gate work with an executable oracle (`validate_corpus`), not a code placeholder; all deterministic code (parse/validate/map, assemble, reweight, driver wiring) is complete and unit-tested with the LLM mocked.
- **Type consistency:** `song_id: str` and `track_row: int` throughout; `CandidateTable` produced by Task 1 is consumed identically by Tasks 3–6; the export columns in Task 4/Task 6 match the Global Constraints schema verbatim; `EngagementModel.sample` / `validate_corpus` signatures match Plan 1.

## Next plan (Plan 3)
Eval: build the 960-d embedding matrix for `train_history.py` from `CandidateTable`, train the predictor on `synth_listening_v3.parquet`, and measure next-track hit-rate / rank on a **held-out slice of the real 3,194 events** vs the retrieval-only (kNN) baseline — the real success criterion for Option A.
