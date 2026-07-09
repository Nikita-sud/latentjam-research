# Synthetic Listening — Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic data foundation the LLM-teacher pipeline depends on — a `Music.UID` reproduction, a full-library track manifest, an engagement-signal model calibrated to real events, and a corpus validation gate.

**Architecture:** Four independent, individually-testable Python modules under `src/synth/`, each with a thin `scripts/` CLI where a runnable entrypoint is useful. They produce the interfaces the later generation/export plan consumes: `song_uid()` (id bridge), `manifest.parquet` (candidate library), `EngagementModel` (label sampler), and `validate_corpus()` (the training gate). No LLM calls in this plan.

**Tech Stack:** Python 3.12 (repo floor 3.11), pandas 2.2, pyarrow, numpy 2.2, stdlib `sqlite3`/`hashlib`/`uuid`, `click` (CLI), pytest (`pythonpath=["src"]`). All already in `pyproject.toml`.

## Global Constraints

- Package layout is **src-layout**: importable code goes in `src/synth/`, tests in `tests/test_synth_*.py`, runnable CLIs in `scripts/`. `pyproject.toml` sets `pythonpath=["src"]` and `testpaths=["tests"]`.
- Run tests with `python -m pytest` from repo root `latentjam-research/`.
- Ruff: `line-length=100`, lint select `E,F,W,I,B,UP,N`. Run `ruff check src/synth tests` before each commit.
- `song_id` everywhere is the canonical `Music.UID` string (e.g. `uas4f3f8624-e9df-c660-def2-f7c4e8dc5753`) — the same id used in `TrackEmbeddingEntity.songUid` and `ListeningEventEntity.songUid`.
- **Authoritative metadata source is `music_cache.db` → `CachedFileData` (814 rows).** Do **not** source the manifest from `data/manifests/personal_606.csv` (stale, 606/810).
- Real-events calibration reference is the fresh DB `db-backups/playback_persistence-2026-07-09.db` (`ListeningEventEntity`, 3,194 rows) or its export `real_listening_events_2026-07-09.csv`.
- New intermediate artifacts go under `data/synth/` (create the dir; git-ignore large parquet if needed).
- Reference for the `Music.UID` algorithm: `latentjam/musikr/src/main/java/org/oxycblt/musikr/Music.kt` (UID = `"u" + formatChar + itemChar + uuid`; AUXIO UUID = `uuid.UUID(bytes=sha256(payload)[:16])`).
- Reference for the output event schema: existing `scripts/synthesize_listening.py` (`listening_event` parquet schema).

---

### Task 1: `Music.UID` reproduction (`src/synth/uid.py`)

Reverse-engineer and reproduce the app's song UID so Python-side ids match on-device ids. This is a **characterization task**: the ground-truth oracle is `data/manifests/personal_resolved.csv`, which pairs real `uas…` ids with track metadata.

**Files:**
- Create: `src/synth/__init__.py`
- Create: `src/synth/uid.py`
- Test: `tests/test_synth_uid.py`
- Read-only reference: `../latentjam/musikr/src/main/java/org/oxycblt/musikr/Music.kt` and the Song model that calls `UID.auxio(this) { update(...) }`.

**Interfaces:**
- Produces:
  - `def song_uid_auxio(*, name: str, album: str | None, artists: list[str], date: str | None, track: int | None, disc: int | None) -> str` — returns the `uas…` string for a song with no MusicBrainz id.
  - `def song_uid_mbid(mbid: str) -> str` — returns the MusicBrainz-format song UID for a recording MBID.
  - `def song_uid(*, mbid: str | None, name: str, album: str | None, artists: list[str], date: str | None = None, track: int | None = None, disc: int | None = None) -> str` — picks the MBID path when `mbid` is present, else auxio.

- [ ] **Step 1: Read the exact hashed fields from the source**

Run and read the output — you need the exact `update(...)` sequence for a Song and the two `microNamespace` chars:
```bash
cd ../latentjam
sed -n '77,240p' musikr/src/main/java/org/oxycblt/musikr/Music.kt          # UID class, Format chars, auxio()/musicBrainz()
grep -rn 'UID.auxio\|UID.musicBrainz\|microNamespace' musikr/src/main/java/org/oxycblt/musikr/model/ musikr/src/main/java/org/oxycblt/musikr/graph/
```
Record: (a) the ordered list of fields hashed for a Song's auxio UID and how each is encoded (UTF-8 bytes; note any lowercasing/normalization or separators), (b) the Song item `microNamespace` char (expected `'s'`), (c) the `MUSICBRAINZ` format `microNamespace` char.

- [ ] **Step 2: Write the failing characterization test**

`tests/test_synth_uid.py`:
```python
import csv
from pathlib import Path

import pytest

from synth.uid import song_uid

# personal_resolved.csv columns: track_id (the uas… UID), path, title, artist, ...
GROUND_TRUTH = Path(__file__).resolve().parents[1].parent / "latentjam-research" / "data" / "manifests" / "personal_resolved.csv"


def _load_rows():
    if not GROUND_TRUTH.exists():
        pytest.skip(f"ground-truth resolver missing: {GROUND_TRUTH}")
    with open(GROUND_TRUTH, newline="") as fh:
        return list(csv.DictReader(fh))


def test_reproduces_known_auxio_uids():
    rows = _load_rows()
    auxio = [r for r in rows if r["track_id"].startswith("uas")]
    assert auxio, "expected at least some auxio-format (uas…) ids in the resolver"
    matched = 0
    for r in auxio:
        got = song_uid(
            mbid=None,
            name=r["title"],
            album=r.get("album") or None,
            artists=[r["artist"]] if r.get("artist") else [],
            date=r.get("date") or None,
            track=int(r["track"]) if r.get("track") else None,
            disc=int(r["disc"]) if r.get("disc") else None,
        )
        matched += got == r["track_id"]
    # Require near-perfect reproduction; a few rows may lack the exact fields the app hashed.
    assert matched / len(auxio) >= 0.99, f"only {matched}/{len(auxio)} auxio ids reproduced"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_synth_uid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'synth.uid'`.

- [ ] **Step 4: Implement `src/synth/uid.py`**

The UUID construction and `toString` format are known and fixed; the only thing you replicate from Step 1 is the exact `_auxio_payload(...)` byte assembly.
```python
"""Reproduce musikr's Music.UID for songs so Python ids match on-device ids.

UID.toString() == "u" + format.microNamespace + item.microNamespace + uuid
(see latentjam/musikr/.../Music.kt). AUXIO songs hash their non-subjective
metadata with SHA-256; the first 16 bytes become the UUID. Java
`UUID(msb, lsb)` over the first 16 big-endian bytes is identical to
`uuid.UUID(bytes=digest[:16])`.
"""

from __future__ import annotations

import hashlib
import uuid

_AUXIO = "a"  # Format.AUXIO.microNamespace
_MUSICBRAINZ = "b"  # Format.MUSICBRAINZ.microNamespace — CONFIRM the exact char in Step 1
_SONG = "s"  # Song item microNamespace — CONFIRM in Step 1


def _uuid_from_sha256(payload: bytes) -> uuid.UUID:
    return uuid.UUID(bytes=hashlib.sha256(payload).digest()[:16])


def _auxio_payload(
    *, name: str, album: str | None, artists: list[str], date: str | None, track: int | None, disc: int | None
) -> bytes:
    """Assemble the exact bytes musikr hashes for a Song's auxio UID.

    Replicate the ordered `update(...)` calls read in Step 1 EXACTLY (field
    order, UTF-8 encoding, any normalization/separators). Iterate against the
    characterization test until it passes.
    """
    h = bytearray()
    h += name.encode("utf-8")
    if album:
        h += album.encode("utf-8")
    for a in artists:
        h += a.encode("utf-8")
    if date:
        h += date.encode("utf-8")
    if track is not None:
        h += track.to_bytes(4, "big", signed=True)
    if disc is not None:
        h += disc.to_bytes(4, "big", signed=True)
    return bytes(h)


def song_uid_auxio(
    *, name: str, album: str | None, artists: list[str], date: str | None = None,
    track: int | None = None, disc: int | None = None,
) -> str:
    u = _uuid_from_sha256(_auxio_payload(name=name, album=album, artists=artists, date=date, track=track, disc=disc))
    return f"u{_AUXIO}{_SONG}{u}"


def song_uid_mbid(mbid: str) -> str:
    return f"u{_MUSICBRAINZ}{_SONG}{uuid.UUID(mbid)}"


def song_uid(
    *, mbid: str | None, name: str, album: str | None, artists: list[str],
    date: str | None = None, track: int | None = None, disc: int | None = None,
) -> str:
    if mbid:
        return song_uid_mbid(mbid)
    return song_uid_auxio(name=name, album=album, artists=artists, date=date, track=track, disc=disc)
```

- [ ] **Step 5: Iterate `_auxio_payload` against the oracle until the test passes**

Run: `python -m pytest tests/test_synth_uid.py -v`
Expected: PASS (≥99% of `uas…` rows reproduced). If it fails, diff one row: print your `song_uid(...)` vs `track_id`, adjust the field order/encoding in `_auxio_payload` to match the Kotlin `update(...)` sequence from Step 1, repeat. Confirm `_MUSICBRAINZ` / `_SONG` chars match Step 1.

- [ ] **Step 6: Commit**

```bash
ruff check src/synth tests/test_synth_uid.py
git add src/synth/__init__.py src/synth/uid.py tests/test_synth_uid.py
git commit -m "feat(synth): reproduce musikr Music.UID for songs"
```

---

### Task 2: Library manifest builder (`src/synth/manifest.py`)

Turn the fresh on-device caches into one clean per-track manifest keyed by `song_id`, covering the full current library.

**Files:**
- Create: `src/synth/manifest.py`
- Create: `scripts/synth_build_manifest.py` (click CLI)
- Test: `tests/test_synth_manifest.py`

**Interfaces:**
- Consumes: `synth.uid.song_uid` (Task 1).
- Produces:
  - `def build_manifest(music_cache_db: str, playback_db: str, audit_csv: str | None = None) -> pandas.DataFrame` with columns `["song_id","title","artist","album","genre","year","language","bpm","energy"]`.
  - CLI writes `data/synth/manifest.parquet`.

- [ ] **Step 1: Write the failing test (with a tiny synthetic sqlite fixture)**

`tests/test_synth_manifest.py`:
```python
import sqlite3
import uuid

import pandas as pd

from synth.manifest import build_manifest
from synth.uid import song_uid


def _make_cache_db(path):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE CachedFileData (uri TEXT PRIMARY KEY, name TEXT, artistNames TEXT, "
        "albumName TEXT, genreNames TEXT, date TEXT, bpm INTEGER, durationMs INTEGER, "
        "musicBrainzId TEXT, track INTEGER, disc INTEGER)"
    )
    # one auxio track (no mbid), one musicbrainz track
    con.execute(
        "INSERT INTO CachedFileData VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("file://a.mp3", "For Leah", "Jake Chudnow", "Singles", "Ambient", "2011", 90, 200000, None, 1, 1),
    )
    mbid = str(uuid.uuid4())
    con.execute(
        "INSERT INTO CachedFileData VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("file://b.mp3", "Song B", "Artist B", "Album B", "Pop", "2020", 120, 180000, mbid, 2, 1),
    )
    con.commit()
    con.close()
    return mbid


def _make_playback_db(path, song_ids):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE TrackEmbeddingEntity (songUid TEXT PRIMARY KEY, modelVersion TEXT, "
        "embedding BLOB, embeddedAtMs INTEGER, tempo REAL, energy REAL)"
    )
    for sid in song_ids:
        con.execute(
            "INSERT INTO TrackEmbeddingEntity VALUES (?,?,?,?,?,?)",
            (sid, "v3", b"\x00", 0, 90.0, 0.5),
        )
    con.commit()
    con.close()


def test_build_manifest_joins_metadata_and_features(tmp_path):
    cache = tmp_path / "music_cache.db"
    playback = tmp_path / "playback.db"
    mbid = _make_cache_db(cache)

    sid_a = song_uid(mbid=None, name="For Leah", album="Singles", artists=["Jake Chudnow"], date="2011", track=1, disc=1)
    sid_b = song_uid(mbid=mbid, name="Song B", album="Album B", artists=["Artist B"], date="2020", track=2, disc=1)
    _make_playback_db(playback, [sid_a, sid_b])

    df = build_manifest(str(cache), str(playback))
    assert list(df.columns) == ["song_id", "title", "artist", "album", "genre", "year", "language", "bpm", "energy"]
    assert set(df["song_id"]) == {sid_a, sid_b}
    row_a = df.set_index("song_id").loc[sid_a]
    assert row_a["title"] == "For Leah" and row_a["artist"] == "Jake Chudnow"
    assert row_a["energy"] == 0.5  # joined from playback db
    assert int(row_a["year"]) == 2011
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_synth_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'synth.manifest'`.

- [ ] **Step 3: Implement `src/synth/manifest.py`**

```python
"""Build the candidate-track manifest from the on-device caches, keyed by Music.UID."""

from __future__ import annotations

import sqlite3

import pandas as pd

from synth.uid import song_uid

_COLUMNS = ["song_id", "title", "artist", "album", "genre", "year", "language", "bpm", "energy"]


def _first(value: str | None) -> str | None:
    """CachedFileData multi-value fields (artistNames/genreNames) are ';'-separated."""
    if not value:
        return None
    return value.split(";")[0].strip() or None


def _year(date: str | None) -> int | None:
    if not date:
        return None
    head = date[:4]
    return int(head) if head.isdigit() else None


def build_manifest(music_cache_db: str, playback_db: str, audit_csv: str | None = None) -> pd.DataFrame:
    cache = pd.read_sql_query("SELECT * FROM CachedFileData", sqlite3.connect(music_cache_db))
    rows = []
    for r in cache.itertuples(index=False):
        artist = _first(r.artistNames)
        rows.append(
            {
                "song_id": song_uid(
                    mbid=(r.musicBrainzId or None),
                    name=r.name,
                    album=r.albumName or None,
                    artists=[a.strip() for a in (r.artistNames or "").split(";") if a.strip()],
                    date=r.date or None,
                    track=int(r.track) if r.track is not None else None,
                    disc=int(r.disc) if r.disc is not None else None,
                ),
                "title": r.name,
                "artist": artist,
                "album": r.albumName or None,
                "genre": _first(r.genreNames),
                "year": _year(r.date),
                "language": None,
                "bpm": float(r.bpm) if r.bpm is not None else None,
                "energy": None,
            }
        )
    man = pd.DataFrame(rows)

    feats = pd.read_sql_query(
        "SELECT songUid AS song_id, tempo, energy FROM TrackEmbeddingEntity", sqlite3.connect(playback_db)
    )
    man = man.merge(feats, on="song_id", how="left")
    # prefer embedded tempo when bpm tag is missing
    man["bpm"] = man["bpm"].where(man["bpm"].notna(), man["tempo"])
    man["energy"] = man["energy_y"] if "energy_y" in man else man["energy"]
    man = man.drop(columns=[c for c in ["tempo", "energy_x", "energy_y"] if c in man.columns], errors="ignore")

    if audit_csv:
        audit = pd.read_csv(audit_csv)
        by_key = audit.assign(_k=(audit["title"].str.lower() + "|" + audit["artist"].str.lower()))
        man_key = (man["title"].str.lower() + "|" + man["artist"].str.lower())
        lang = by_key.set_index("_k")["language"].to_dict()
        genre = by_key.set_index("_k")["genre"].to_dict()
        man["language"] = man_key.map(lang).where(man["language"].isna(), man["language"])
        man["genre"] = man["genre"].where(man["genre"].notna(), man_key.map(genre))

    return man[_COLUMNS]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_synth_manifest.py -v`
Expected: PASS.

- [ ] **Step 5: Add the CLI `scripts/synth_build_manifest.py`**

```python
"""Build data/synth/manifest.parquet from the synced on-device caches."""

from pathlib import Path

import click

from synth.manifest import build_manifest


@click.command()
@click.option("--music-cache", required=True, type=click.Path(exists=True), help="music_cache.db from the phone")
@click.option("--playback-db", required=True, type=click.Path(exists=True), help="playback_persistence-*.db")
@click.option("--audit-csv", default="data/manifests/music_audit_full_tags.csv", type=click.Path())
@click.option("--out", default="data/synth/manifest.parquet", type=click.Path())
def main(music_cache: str, playback_db: str, audit_csv: str, out: str) -> None:
    df = build_manifest(music_cache, playback_db, audit_csv if Path(audit_csv).exists() else None)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    click.echo(f"wrote {len(df)} tracks -> {out}  (missing genre: {df['genre'].isna().sum()})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-run against the real caches and eyeball coverage**

Run:
```bash
python scripts/synth_build_manifest.py \
  --music-cache "$SCRATCH/phone-sync-2026-07-09/music_cache.db" \
  --playback-db ../db-backups/playback_persistence-2026-07-09.db
```
Expected: `wrote ~810 tracks -> data/synth/manifest.parquet`, with most rows carrying a non-null `energy` (embeddings present) — confirms the `song_id` bridge lines up with `TrackEmbeddingEntity`. If `energy` is null for most rows, the UID reproduction (Task 1) is off — go back before proceeding.

- [ ] **Step 7: Commit**

```bash
ruff check src/synth/manifest.py scripts/synth_build_manifest.py tests/test_synth_manifest.py
git add src/synth/manifest.py scripts/synth_build_manifest.py tests/test_synth_manifest.py
git commit -m "feat(synth): build library manifest keyed by Music.UID"
```

---

### Task 3: Engagement calibration model (`src/synth/engagement.py`)

Learn the real engagement marginals (completion rate, finalize-reason mix, skip-position and played-fraction distributions) from the 3,194 real events, and expose a sampler that reproduces them. The LLM will supply track *ordering*; this model supplies *labels* — so they aren't implausibly clean.

**Files:**
- Create: `src/synth/engagement.py`
- Test: `tests/test_synth_engagement.py`

**Interfaces:**
- Produces:
  - `def load_real_events(db_or_csv: str) -> pandas.DataFrame` (reads `ListeningEventEntity` from a sqlite db, or a CSV export).
  - `def calibration_targets(events: pandas.DataFrame) -> dict` → `{"completion_rate": float, "finalize_reason": {reason: prob}, "played_fraction_completed": (mean,std), "played_fraction_skipped": (mean,std)}`.
  - `@dataclass EventLabels(played_fraction: float, skipped: bool, completed: bool, finalize_reason: str)`.
  - `class EngagementModel` with `@classmethod from_events(cls, events) -> EngagementModel` and `def sample(self, session_len: int, rng: numpy.random.Generator) -> list[EventLabels]` (skip probability rises with position so skips cluster early).

- [ ] **Step 1: Write the failing test**

`tests/test_synth_engagement.py`:
```python
import numpy as np
import pandas as pd

from synth.engagement import EngagementModel, calibration_targets


def _fake_events(n=4000, completion=0.44, seed=0):
    rng = np.random.default_rng(seed)
    completed = rng.random(n) < completion
    return pd.DataFrame(
        {
            "completed": completed.astype(int),
            "skipped": (~completed).astype(int),
            "playedMs": np.where(completed, 200000, rng.integers(1000, 60000, n)),
            "trackDurationMs": 210000,
            "sessionPos": rng.integers(0, 15, n),
            "finalizeReason": rng.choice(
                ["TRACK_ENDED", "NEW_PLAYBACK", "USER_SKIPPED", "SESSION_END"],
                n, p=[0.42, 0.32, 0.25, 0.01],
            ),
        }
    )


def test_calibration_targets_match_input():
    ev = _fake_events()
    t = calibration_targets(ev)
    assert abs(t["completion_rate"] - 0.44) < 0.03
    assert set(t["finalize_reason"]) == {"TRACK_ENDED", "NEW_PLAYBACK", "USER_SKIPPED", "SESSION_END"}
    assert abs(sum(t["finalize_reason"].values()) - 1.0) < 1e-6


def test_sampled_corpus_reproduces_completion_rate():
    ev = _fake_events()
    model = EngagementModel.from_events(ev)
    rng = np.random.default_rng(1)
    labels = [lbl for _ in range(500) for lbl in model.sample(session_len=10, rng=rng)]
    got = np.mean([lbl.completed for lbl in labels])
    assert abs(got - 0.44) < 0.04
    # completed rows carry a high played fraction; skipped rows a low one
    comp = [lbl.played_fraction for lbl in labels if lbl.completed]
    skip = [lbl.played_fraction for lbl in labels if lbl.skipped]
    assert np.mean(comp) > 0.85 and np.mean(skip) < 0.5
    # every label is internally consistent
    assert all(lbl.completed != lbl.skipped for lbl in labels)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_synth_engagement.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'synth.engagement'`.

- [ ] **Step 3: Implement `src/synth/engagement.py`**

```python
"""Engagement-signal model calibrated to real ListeningEventEntity marginals."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

_COMPLETION_THRESHOLD = 0.9  # app defines completed as playedMs >= 0.9 * durationMs


def load_real_events(db_or_csv: str) -> pd.DataFrame:
    if db_or_csv.endswith(".csv"):
        return pd.read_csv(db_or_csv)
    return pd.read_sql_query("SELECT * FROM ListeningEventEntity", sqlite3.connect(db_or_csv))


def calibration_targets(events: pd.DataFrame) -> dict:
    frac = (events["playedMs"] / events["trackDurationMs"]).clip(0, 1)
    completed = events["completed"].astype(bool)
    reasons = events["finalizeReason"].value_counts(normalize=True).to_dict()
    return {
        "completion_rate": float(completed.mean()),
        "finalize_reason": {str(k): float(v) for k, v in reasons.items()},
        "played_fraction_completed": (float(frac[completed].mean()), float(frac[completed].std() or 0.05)),
        "played_fraction_skipped": (float(frac[~completed].mean()), float(frac[~completed].std() or 0.15)),
    }


@dataclass
class EventLabels:
    played_fraction: float
    skipped: bool
    completed: bool
    finalize_reason: str


class EngagementModel:
    def __init__(self, targets: dict):
        self._t = targets
        reasons = targets["finalize_reason"]
        self._reason_names = list(reasons)
        self._reason_probs = np.array([reasons[r] for r in self._reason_names])
        self._reason_probs = self._reason_probs / self._reason_probs.sum()

    @classmethod
    def from_events(cls, events: pd.DataFrame) -> "EngagementModel":
        return cls(calibration_targets(events))

    def _skip_prob(self, pos: int) -> float:
        """Base skip rate, tilted so skips cluster early in a session."""
        base = 1.0 - self._t["completion_rate"]
        tilt = 1.25 if pos < 3 else (0.85 if pos >= 6 else 1.0)
        return float(np.clip(base * tilt, 0.02, 0.98))

    def sample(self, session_len: int, rng: np.random.Generator) -> list[EventLabels]:
        out: list[EventLabels] = []
        for pos in range(session_len):
            skipped = rng.random() < self._skip_prob(pos)
            completed = not skipped
            if completed:
                mean, std = self._t["played_fraction_completed"]
            else:
                mean, std = self._t["played_fraction_skipped"]
            frac = float(np.clip(rng.normal(mean, std), 0.0, 1.0))
            if skipped:
                reason = "USER_SKIPPED"
            else:
                reason = str(rng.choice(self._reason_names, p=self._reason_probs))
                if reason == "USER_SKIPPED":  # completed rows never carry USER_SKIPPED
                    reason = "TRACK_ENDED"
            out.append(EventLabels(frac, skipped, completed, reason))
        return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_synth_engagement.py -v`
Expected: PASS.

- [ ] **Step 5: Sanity-check the targets against the real data**

Run:
```bash
python -c "from synth.engagement import load_real_events, calibration_targets; import json; \
print(json.dumps(calibration_targets(load_real_events('../db-backups/playback_persistence-2026-07-09.db')), indent=2))"
```
Expected: `completion_rate ≈ 0.44`, `finalize_reason` dominated by `TRACK_ENDED`. If it reports ~0.19, you loaded the June `pre-purge` db by mistake — use the July one.

- [ ] **Step 6: Commit**

```bash
ruff check src/synth/engagement.py tests/test_synth_engagement.py
git add src/synth/engagement.py tests/test_synth_engagement.py
git commit -m "feat(synth): engagement model calibrated to real events"
```

---

### Task 4: Corpus validation gate (`src/synth/validate.py`)

Compute distribution-match and coverage metrics for a generated corpus against the real reference, and return a hard PASS/FAIL. This is the training gate.

**Files:**
- Create: `src/synth/validate.py`
- Test: `tests/test_synth_validate.py`

**Interfaces:**
- Consumes: a `manifest` DataFrame (Task 2) and a real-events DataFrame (Task 3).
- Produces:
  - `def kl_divergence(p: dict, q: dict) -> float` (over a shared support, with Laplace smoothing).
  - `@dataclass ValidationReport(passed: bool, metrics: dict, failures: list[str])`.
  - `def validate_corpus(sessions: pandas.DataFrame, manifest: pandas.DataFrame, real_events: pandas.DataFrame, *, max_genre_kl: float = 0.25, min_coverage: float = 0.6, max_completion_delta: float = 0.05) -> ValidationReport`. `sessions` has one row per generated event with columns `["song_id","completed"]` (extra columns ignored).

- [ ] **Step 1: Write the failing test**

`tests/test_synth_validate.py`:
```python
import numpy as np
import pandas as pd

from synth.validate import kl_divergence, validate_corpus


def _manifest():
    genres = (["Anime OST"] * 5 + ["Hip-Hop"] * 3 + ["Disco"] * 2)
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_synth_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'synth.validate'`.

- [ ] **Step 3: Implement `src/synth/validate.py`**

```python
"""Corpus validation gate: distribution match + coverage vs the real reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _dist(values: pd.Series, support: list[str]) -> dict:
    counts = values.value_counts().to_dict()
    total = sum(counts.get(k, 0) for k in support) + len(support)  # Laplace smoothing
    return {k: (counts.get(k, 0) + 1) / total for k in support}


def kl_divergence(p: dict, q: dict) -> float:
    support = sorted(set(p) | set(q))
    pa = np.array([p.get(k, 1e-9) for k in support])
    qa = np.array([q.get(k, 1e-9) for k in support])
    pa, qa = pa / pa.sum(), qa / qa.sum()
    return float(np.sum(pa * np.log(pa / qa)))


@dataclass
class ValidationReport:
    passed: bool
    metrics: dict
    failures: list[str]


def validate_corpus(
    sessions: pd.DataFrame,
    manifest: pd.DataFrame,
    real_events: pd.DataFrame,
    *,
    max_genre_kl: float = 0.25,
    min_coverage: float = 0.6,
    max_completion_delta: float = 0.05,
) -> ValidationReport:
    genre_of = manifest.set_index("song_id")["genre"].to_dict()
    support = sorted({g for g in genre_of.values() if isinstance(g, str)})

    syn_genres = sessions["song_id"].map(genre_of).dropna()
    real_genres = real_events["song_id"].map(genre_of).dropna()
    genre_kl = kl_divergence(_dist(syn_genres, support), _dist(real_genres, support))

    coverage = sessions["song_id"].nunique() / max(len(manifest), 1)
    completion_delta = abs(sessions["completed"].mean() - real_events["completed"].mean())

    metrics = {
        "genre_kl": genre_kl,
        "coverage": coverage,
        "completion_delta": completion_delta,
        "n_sessions_events": int(len(sessions)),
    }
    failures = []
    if genre_kl > max_genre_kl:
        failures.append(f"genre_kl {genre_kl:.3f} > {max_genre_kl}")
    if coverage < min_coverage:
        failures.append(f"coverage {coverage:.3f} < {min_coverage}")
    if completion_delta > max_completion_delta:
        failures.append(f"completion_delta {completion_delta:.3f} > {max_completion_delta}")
    return ValidationReport(passed=not failures, metrics=metrics, failures=failures)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_synth_validate.py -v`
Expected: PASS (all three tests).

- [ ] **Step 5: Full suite + lint**

Run: `python -m pytest tests/test_synth_*.py -v && ruff check src/synth tests`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/synth/validate.py tests/test_synth_validate.py
git commit -m "feat(synth): corpus validation gate (genre KL, coverage, completion delta)"
```

---

## Self-Review

- **Spec coverage:** Task 1 ⇒ spec §5 Stage 1 "songUid bridge"; Task 2 ⇒ Stage 1 manifest; Task 3 ⇒ Stage 5 engagement derivation + §2 calibration targets; Task 4 ⇒ Stage 6 validation gate. Stages 2 (lyrics — deferred to v3.1 per §10), 3 (persona/goal grid), 4 (LLM generation), 7 (export) are **out of scope for this plan** and belong to the next plan, which consumes `manifest.parquet`, `EngagementModel`, and `validate_corpus` produced here. Popularity-bias reweighting (Stage 6) is deferred with generation (it acts on generated candidate sampling).
- **Placeholder scan:** the only intentionally-open item is the exact `_auxio_payload` field order in Task 1 — resolved by reading the source in Step 1 against an executable ground-truth oracle (not a placeholder: it has a pass/fail test). `_MUSICBRAINZ`/`_SONG` chars flagged CONFIRM with the same source read.
- **Type consistency:** `song_uid(...)` signature is identical in Task 1 (definition), Task 2 (import), and the manifest test. `EngagementModel.from_events`/`sample`, `EventLabels`, `ValidationReport`, `validate_corpus(...)` names match between their defining task and their tests. Manifest columns `["song_id","title","artist","album","genre","year","language","bpm","energy"]` are asserted identically in Task 2's test and consumed by Task 4 (`song_id`, `genre`).

## Next plan (not this one)
Generation + export: persona×goal grid, Ollama/vLLM guided-JSON session generation over `manifest.parquet` (songId enum), applying `EngagementModel` for labels, popularity-bias reweighting, running `validate_corpus` as the gate, exporting `models/predictor/synth_listening_v3.parquet` for `train_history.py`. Then the downstream eval: train on v3, measure held-out hit-rate on the real 3,194 events vs the retrieval baseline.
