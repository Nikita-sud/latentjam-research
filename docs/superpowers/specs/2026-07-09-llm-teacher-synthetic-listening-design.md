# LLM-Teacher Synthetic Listening Corpus — Design

- **Date:** 2026-07-09
- **Status:** Reviewed — decisions locked 2026-07-09
- **Owner:** Nikita
- **Repo:** `latentjam-research` (offline generation + training). Consumer: the on-device predictor in `latentjam`.

## 1. Goal

Pretrain the on-device next-track predictor from a **synthetic listening corpus** generated offline by a **local LLM teacher**, so the predictor is useful on day-1 instead of waiting for real interaction data to accumulate. This is "Option A": a one-time, developer-side generation over a representative library, shipping pretrained weights; on-device fine-tuning (when `RetrainWorker` becomes real) adapts per-user later.

The LLM sees **track metadata only** (title, artist, album, genre, year, bpm, energy, optional lyric-derived tags) — never audio. Its job is **contextual/cultural coherence**: which artists/genres/moods co-occur, plausible session arcs (workout / focus / evening chill / anime binge), and realistic skip behavior. Output is structured JSON, generated in bulk, then converted to `(history → next-track)` training pairs for `train_history.py`.

## 2. Why this is needed (grounding facts)

- **Real data is far too sparse to train from scratch.** Fresh device DB (`db-backups/playback_persistence-2026-07-09.db`, pulled 2026-07-09): **810 tracks, 3,194 events, only 69 sessions** over 2026-05-27 → 2026-07-09.
- **The learned predictor is currently inert** (assets absent, `predictorEnabled=false`, dim-mismatched 512 vs 960); production is kNN retrieval + `MetadataRerank`. A prior learned scorer (`scoring_v1.pt`) overfit catastrophically on personal-playlist data — synthetic bias is the central risk to manage.
- **The library is eclectic and non-Western**, exactly where an LLM's cultural knowledge beats co-occurrence datasets (MPD). Top genres (of 746 tagged): Anime OST 54, Hip-Hop 50, Film Score 32, Pop Rock 30, Disco 29, Russian Pop 25, Dance-pop 22, Hard Rock 20, Game OST 19, Eurodance 18, Russian Rock 16, Synth-pop 15. Much of it is instrumental (Anime/Film/Game OST).
- **Real engagement priors (calibration targets, from the fresh DB):** completion **44.4%** (1419 completed / 1775 skipped), `finalizeReason` mix TRACK_ENDED 1326 / NEW_PLAYBACK 1035 / USER_SKIPPED 810 / SESSION_END 23, smart-picks 1658/3194. (The June `pre-purge` backup showed 19% completion — inflated by sub-second queue-churn the newer build now filters; **calibrate against the July data, not June**.)

## 3. Scope

**In scope:** offline pipeline that turns the library manifest into a validated synthetic listening corpus in the `ListeningEventEntity` shape, ready for `train_history.py`.

**Out of scope (explicit non-goals):**
- On-device generation or any per-user LLM calls in the shipped app.
- Fixing the predictor architecture / 512-vs-960 mismatch and re-enabling it in the app (tracked separately — this design produces the *data* that pretraining needs).
- Re-encoding audio (embeddings already exist per track).

## 4. Decisions locked (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Approach | Option A: offline generic pretraining | Real blocker is "no learner + no data", not "not enough compute"; keeps privacy ethos (nothing per-user shipped to an API). |
| Bulk teacher | **Local** — `qwen3.6:35b` (Ollama), `aya-expanse:32b` for rus/anime long-tail | Already downloaded; fits M5 Max 64 GB; `guided_json`/GBNF = hard schema guarantee; Aya is multilingual (strong on Russian). |
| Not chosen | DeepSeek-V4 | Won't fit 64 GB; its edge is *factual* music QA, not the *contextual* coherence we need; Aya likely stronger on the specific rus long-tail. |
| Compute | Mac for prototype+validation; **RunPod H100 (vLLM) only for large bulk**, behind a validation gate | Mac single-stream throughput makes 100k sessions impractical (~weeks); an H100 does ~150M tokens in a few GPU-hours (~$10–60). Rent only if the target count is large and the pilot passes validation. |
| Gold-seed / judge | Optional small Claude slice (`claude-opus-4-8`) | A few hundred–few thousand exemplars used as in-context few-shot to lift the local model on the long-tail; optional, only if validation shows blandness. |
| Lyrics | Optional, **derived tags only**, local-only | Adds semantic signal orthogonal to audio embeddings; local generation means copyrighted text never leaves the machine and never enters the output corpus. |

## 5. Architecture

Seven stages, each independently runnable and testable, communicating via files on disk (CSV/Parquet/JSONL).

```
(1) build_library_manifest ──► manifest.parquet  (per-track: id, title, artist, album, genre, year, bpm, energy [, lyric tags])
(2) extract_lyric_features  ──► lyric_tags.parquet  (optional; vocal subset only)
(3) build_persona_goal_grid ──► grid.jsonl  (persona dials × intent goals, weighted)
(4) generate_sessions (LLM) ──► sessions_raw.jsonl  (ordered songId sequences + per-item "fit" rationale; NO engagement labels)
(5) derive_engagement       ──► sessions_labeled.jsonl  (playedFraction / skipped / completed / finalizeReason from a calibrated statistical model)
(6) validate_corpus (gate)  ──► report + PASS/FAIL  (KL, coverage, popularity-bias; reweight)
(7) export_trainpairs       ──► synth_listening_v3.parquet  (feeds train_history.py)
```

### Stage 1 — Library manifest
Build one clean per-track record covering the **full current library (810 tracks)**.

- **Authoritative metadata source:** `music_cache.db` → `CachedFileData` (814 rows, freshest, keyed by `uri`; carries `name`/`artistNames`/`albumName`/`genreNames`/`date`/`bpm`/`durationMs`/`musicBrainzId`). **Do not use `data/manifests/personal_606.csv`/`personal_resolved.csv` as the source — they are stale (606/810 tracks, ~200 missing).**
- **songUid bridge:** `CachedFileData` is keyed by `uri`, but events/embeddings are keyed by `songUid` (`Music.UID`). Reproduce the app's `Music.UID` computation (defined in the `musikr` module of `latentjam`; uses the MusicBrainz ID when present, else a metadata hash) to derive `songId` per `CachedFileData` row. Cross-check the reproduced mapping against `personal_resolved.csv` on the overlap to confirm correctness before trusting it.
- **Audio features:** join `tempo`/`energy` from `TrackEmbeddingEntity` (fresh DB) by `songUid`.
- **Enrichment:** where `genreNames`/`date`/`language` are sparse in `CachedFileData`, backfill from `music_audit_full_tags.csv` matched on MusicBrainz ID or `(title, artist)`.

Output columns: `songId` (canonical `Music.UID`), `title`, `artist`, `album`, `genre`, `year`, `language`, `bpm`, `energy`, and (later) lyric tags. `songId` is the enum domain for generation.

### Stage 2 — Lyric features (optional)
For the **vocal subset only** (skip Anime/Film/Game OST instrumentals): fetch once from LRCLIB (free, has synced lyrics) with Genius fallback; check embedded ID3 `USLT` / `.lrc` first. **Do not store or emit raw lyrics.** Distill each to compact tags: `language`, 2–3 `theme` keywords, `mood`, `explicit` flag, one-line summary. These join into the manifest as extra grounding fields. Local-only.

### Stage 3 — Persona × goal grid
Pre-structure diversity (do **not** rely on sampling temperature — TalkPlayData 2 shows goal-conditioning ablation causes measurable diversity collapse). Each generated session is conditioned on:
- a **persona** with dials (activity level, conformity, diversity-appetite — Agent4Rec-style), and
- an **intent goal** drawn from a topic × specificity grid tailored to this library: `workout`, `focus/study`, `evening chill`, `anime binge`, `russian throwback`, `disco/eurodance party`, `film-score ambient`, `discovery/long-tail`, etc.

Grid is weighted so the mix roughly reflects plausible real usage (and deliberately over-samples the long-tail to counter head bias).

### Stage 4 — LLM session generation
- **Model:** `qwen3.6:35b` primary; route sessions whose seed/goal is rus/anime-heavy to `aya-expanse:32b`. Optional Claude `claude-opus-4-8` gold-seed exemplars as few-shot.
- **Guided decoding:** Ollama `format`=JSON-schema (or LM Studio structured output / vLLM `guided_json` on RunPod). `songId` is a **strict enum** of manifest IDs → zero hallucinated tracks.
- **Prompt:** cached fixed prefix = full candidate library (id + compact metadata) + schema + persona/goal instructions; volatile suffix = this batch's persona/goal draws. Emit **10–20 sessions per call** behind the cached prefix; compact schema (short field names, no whitespace).
- **What the LLM emits:** an ordered `songId` sequence per session + a short per-item `fit` rationale (why this track fits persona+goal+position). **It does NOT emit engagement labels** (see Stage 5).

### Stage 5 — Engagement-signal derivation (critical)
Per AAAI-2025 "LLM-Powered User Simulator": use the LLM for *logical preference ordering*, then **derive** engagement signals from a persona-conditioned statistical model, **calibrated to the real 3,194 events**. Concretely, sample `skipped`/`completed`, `playedFraction`, and `finalizeReason` so the corpus matches the real marginals (≈44% completion; TRACK_ENDED/NEW_PLAYBACK/USER_SKIPPED/SESSION_END mix) and plausible position/persona effects (skips cluster early; conformist personas complete more; discovery goals skip more). Fill the rest of the `ListeningEventEntity` schema (`sessionPos`, `ctxUid0..3`, `shuffleMode`, timing) deterministically. Rationale: naive LLM-emitted labels are implausibly clean and would teach the encoder a degenerate skip prior.

### Stage 6 — Corpus validation gate
Compute against a real reference (the 3,194 events + library distribution) **before** any training:
- KL-divergence of genre / artist / item-frequency distributions (synthetic vs real).
- Coverage: fraction of library appearing; long-tail representation; no persona/genre mode-collapse.
- Engagement marginals match the calibration targets.
- **Popularity-bias reweighting:** reweight candidate sampling and/or post-hoc reweight sessions so item frequency matches real catalog demand (or deliberately upweights long-tail). LLMs amplify head bias; without this the student inherits the teacher's popularity prior.

Gate is **hard**: training consumes the corpus only if metrics pass. If the local model fails on long-tail blandness, escalate (gold-seed exemplars, or RunPod + larger model), don't ship it.

### Stage 7 — Export to training format
Emit `models/predictor/synth_listening_v3.parquet` in the same shape `train_history.py` already consumes (mirrors `synth_listening_v1/v2.parquet`), reconstructing `TrainPair` tuples from the labeled sessions. No re-encoding — embeddings are looked up by `songId`.

## 6. Compute plan (Mac → RunPod)

**Be realistic about Mac throughput:** M5 Max single-stream on a 35B model is ~30–40 tok/s. A real corpus (20–30k sessions ≈ 30–45M output tokens) would take **>1 week** on the Mac — so the Mac is a **pilot machine, not a bulk machine**.

1. **Pilot + calibrate on the Mac (~1–2k sessions).** `qwen3.6:35b` via Ollama structured-output. Build & fit the engagement model against the 3,194 real events; run the validation gate; debug the whole pipeline. Free, private.
2. **First real corpus on RunPod: ~20–30k sessions.** H100 + vLLM `guided_json`, same validated model (or scale up to Qwen3-235B). One-time, ~$15–40. Data stays on a pod you control (no model-provider API). 20–30k richly covers the persona×goal grid for an 810-track library without overkill.
3. **Scale to 50–100k only if the downstream learning curve is still climbing** (held-out hit-rate on real events keeps improving with more data). Otherwise stop — data is cheap but not free of diminishing returns.

## 7. Reuse of existing assets
- `scripts/synthesize_listening.py` — existing heuristic synth; the LLM pipeline supersedes/augments it (keep as a baseline for the validation gate).
- `scripts/mpd_to_listening_events.py` + MPD sessions — complementary *real-world* co-occurrence data; can seed personas or serve as an additional validation reference.
- `train_history.py`, `models/predictor/synth_listening_v1/v2.parquet` — the trainer and prior corpora; our output is `v3` in the same format.

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Popularity/head-bias distillation (highest impact) | candidate reweighting + post-hoc corpus reweighting + coverage gate |
| Diversity / mode collapse | pre-structured persona×goal grid; KL/coverage gate |
| Unrealistic engagement labels | derive signals statistically, calibrate to the 3,194 real events (Stage 5) |
| Non-Western/long-tail blandness | route rus/anime to Aya; optional Claude gold-seed exemplars; extra judge review on long-tail |
| Schema conformance at scale | guided decoding (Ollama/LM Studio/vLLM) makes malformed JSON impossible + songId enum |
| Teacher bias distilled into student generally | validation gate + human spot-check + keep real-data holdout for evaluation |

## 9. Validation / testing
- **Unit:** each stage has a smoke test on a tiny fixture (10 tracks → 5 sessions → labeled → exported).
- **Corpus metrics:** the Stage 6 gate (KL, coverage, engagement marginals) is the acceptance test.
- **Downstream:** train the predictor on `v3` and measure on a **held-out slice of the real 3,194 events** (the only ground truth) — next-track hit-rate / rank vs the retrieval-only baseline. This is the real success criterion: does synthetic pretraining beat pure kNN on real held-out behavior?

## 10. Decisions (resolved 2026-07-09)
- **Target corpus size:** pilot ~1–2k sessions on the Mac (pipeline build + calibration), then **~20–30k as the first real corpus on RunPod**; scale to 50–100k only if the held-out learning curve is still climbing. The Mac does **not** do the bulk run.
- **songId ↔ metadata join:** `music_cache.db`/`CachedFileData` (814 tracks) is authoritative; bridge to `songId` by reproducing the `Music.UID` computation from the `musikr` module; cross-check against `personal_resolved.csv`. The stale `personal_606.csv` is **not** the source.
- **Lyrics:** **v1 ships without lyrics.** Add lyric-derived tags as a separate **v3.1 ablation** and keep them only if they measurably lift held-out hit-rate.
- **Gold-seed / Claude:** **local-only by default (no Claude).** Add `claude-opus-4-8` in-context exemplars **only if** the validation gate or held-out eval shows the local models are too bland/biased on the rus/anime long-tail (Aya is the first line of defence there).

## 11. Milestones
1. **M1 — Manifest + grid:** Stage 1 (+2 optional), Stage 3. Deliver `manifest.parquet`, `grid.jsonl`.
2. **M2 — Mac pilot + calibration:** Stages 4–5 on the Mac; fit engagement model to the 3,194 events; ~1–2k pilot sessions.
3. **M3 — Validation gate:** Stage 6; iterate persona/goal + reweighting until PASS.
4. **M4 — Bulk generation:** Mac overnight or RunPod H100 depending on target size.
5. **M5 — Train & evaluate:** Stage 7 → `train_history.py` → hit-rate on held-out real events vs retrieval baseline.
