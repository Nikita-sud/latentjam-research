"""Generate the v3 synthetic listening corpus end-to-end.

Pipeline (spec §6): manifest -> candidates -> engagement (calibrated on real
events) -> persona x goal grid -> per-session (candidate subset -> LLM order
-> engagement-labeled assembly) -> popularity reweighting -> validation gate
-> export.

Usage:

    python scripts/synth_generate_corpus.py \\
        --music-cache /path/to/music_cache.db \\
        --playback-db /path/to/playback_persistence-*.db \\
        --n-sessions 200 --model qwen3.6:35b --seed 0
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import pandas as pd

from synth.assemble import assemble_session
from synth.candidates import CandidateTable, build_candidate_table
from synth.engagement import EngagementModel, load_real_events
from synth.generate import generate_session
from synth.manifest import build_manifest
from synth.personas import SessionSpec, build_grid
from synth.reweight import candidate_subset, reweight_sessions
from synth.validate import validate_corpus

_EXPORT_COLUMNS = [
    "user_id", "session_id", "ts_unix_ms", "track_id", "track_row", "played_seconds",
    "track_duration_s", "completed", "skipped", "liked", "context_track_ids",
    "hour_of_day", "day_of_week", "is_weekend", "session_pos",
]

# Rolling gap (ms) between one synthetic session's start and the next's, so
# sessions don't all collide on the same timestamp (irrelevant to training,
# but keeps the corpus's ts_unix_ms monotonic/plausible for inspection).
_SESSION_GAP_MS = 3_600_000  # 1 hour
_BASE_TS_MS = 1_700_000_000_000  # arbitrary fixed epoch anchor (Nov 2023)

GenerateFn = Callable[[SessionSpec, CandidateTable, list[int]], list[str]]


def generate_corpus(
    candidates: CandidateTable,
    engagement: EngagementModel,
    grid: list[SessionSpec],
    *,
    model: str,
    k: int,
    rng: np.random.Generator,
    generate_fn: GenerateFn,
) -> pd.DataFrame:
    """Pure per-session pipeline: subset -> LLM order -> assemble -> concat.

    ``generate_fn`` is injected (real Ollama call in the CLI, a stub in
    tests) so this stays hermetically testable. A single session's failure
    -- the LLM returning malformed/empty output, or hallucinating a
    ``song_id`` the assembler can't resolve -- must not kill the whole run,
    so both ``generate_fn`` and ``assemble_session`` are wrapped per-session
    and a failing session is simply skipped.
    """
    del model  # threaded through generate_fn by the caller; unused here
    all_rows: list[dict] = []
    for i, spec in enumerate(grid):
        subset = candidate_subset(candidates, spec, k=k, rng=rng)
        start_ts_ms = _BASE_TS_MS + i * _SESSION_GAP_MS
        try:
            song_ids = generate_fn(spec, candidates, subset)
            if not song_ids:
                continue
            rows = assemble_session(
                song_ids, spec, candidates, engagement,
                user_id=f"{spec.persona.name}-{i}", session_id=f"synth-{i}",
                start_ts_ms=start_ts_ms, rng=rng,
            )
        except Exception:
            # Task 3's generate can raise on malformed LLM JSON; Task 4's
            # assemble KeyErrors on a hallucinated song_id. Either way: skip
            # this one session, keep the run going.
            continue
        all_rows.extend(rows)
    return pd.DataFrame(all_rows, columns=_EXPORT_COLUMNS)


@click.command()
@click.option(
    "--music-cache",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="music_cache.db synced from the phone (contains CachedFileData).",
)
@click.option(
    "--playback-db",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="playback_persistence-*.db (contains TrackEmbeddingEntity + ListeningEventEntity).",
)
@click.option("--n-sessions", default=200, show_default=True, type=int, help="Sessions to generate.")
@click.option("--model", default="qwen3.6:35b", show_default=True, help="Ollama model tag.")
@click.option(
    "--audit-csv",
    default="data/manifests/music_audit_full_tags.csv",
    type=click.Path(dir_okay=False),
    help="Optional audited-tags CSV used to backfill null genre/language.",
)
@click.option(
    "--out",
    default="models/predictor/synth_listening_v3.parquet",
    type=click.Path(dir_okay=False),
    help="Output parquet path (only written if the validation gate passes).",
)
@click.option("--seed", default=0, show_default=True, type=int, help="RNG seed (grid, subsets, LLM, engagement).")
@click.option("--k", default=120, show_default=True, type=int, help="Candidate subset size per session.")
def main(
    music_cache: str, playback_db: str, n_sessions: int, model: str,
    audit_csv: str, out: str, seed: int, k: int,
) -> None:
    rng = np.random.default_rng(seed)

    audit = audit_csv if Path(audit_csv).exists() else None
    manifest = build_manifest(music_cache, playback_db, audit)
    candidates = build_candidate_table(manifest, playback_db)
    real_events = load_real_events(playback_db)
    engagement = EngagementModel.from_events(real_events)

    grid = build_grid(n_sessions, rng)

    def real_generate_fn(spec: SessionSpec, cand: CandidateTable, subset: list[int]) -> list[str]:
        return generate_session(spec, cand, model=model, subset=subset, seed=seed)

    df = generate_corpus(
        candidates, engagement, grid, model=model, k=k, rng=rng, generate_fn=real_generate_fn
    )
    n_generated = df["session_id"].nunique()
    click.echo(f"generated {n_generated}/{n_sessions} sessions ({len(df)} events)")

    df = reweight_sessions(df, target_freq=None, rng=rng)
    n_reweighted = df["session_id"].nunique()
    click.echo(f"reweighted -> {n_reweighted} sessions ({len(df)} events)")

    sessions_view = df[["track_id", "completed"]].rename(columns={"track_id": "song_id"})
    report = validate_corpus(sessions_view, manifest, real_events)

    if not report.passed:
        click.echo("VALIDATION GATE FAILED — not writing the parquet.", err=True)
        for failure in report.failures:
            click.echo(f"  - {failure}", err=True)
        click.echo(f"metrics: {report.metrics}", err=True)
        raise SystemExit(1)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    click.echo(f"VALIDATION GATE PASSED — wrote {len(df)} events -> {out_path}")
    click.echo(f"  sessions: {df['session_id'].nunique()}, tracks: {df['track_id'].nunique()}")
    click.echo(f"  metrics: {report.metrics}")


if __name__ == "__main__":
    main()
