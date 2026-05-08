# latentjam-research

ML research workspace for **latentjam** — a privacy-first local Android music player. The Android app lives in the sibling repo at `~/Documents/LJ/latentjam/`.

This repo is laptop-only: it builds embedding stores, evaluates models, and exports ONNX artifacts the app will eventually consume. Nothing in here ships to a phone directly. See `CLAUDE.md` for conventions and the pinned encoder recipe.

## v0: content-based similarity

End-to-end pipeline:

1. **Embed** every audio file in a folder using a pretrained music encoder (MERT-v1-95M).
2. **Store** the 768-d embeddings in a single Parquet file alongside metadata.
3. **Recommend** by cosine top-k.
4. **Export** the encoder to ONNX (FP32 + INT8 dynamic quantization).
5. **Verify** numerical equivalence and **benchmark** CPU latency.
6. **Evaluate** retrieval quality on FMA-small (genre recall@k) and MagnaTagATune (tag-Jaccard@k).

No learned predictor, no listening-history features, no Android integration yet. Those are v0.5+.

## Quickstart

```bash
# 1. install the project + dev deps in a fresh virtualenv
python -m venv .venv && source .venv/bin/activate
make install

# 2. embed your local FLAC collection (or any folder of audio files)
make embed ROOT=/path/to/music OUT=models/store/library.parquet

# 3. ask for top-10 similar tracks given a track_id (xxhash from the embed step)
make recommend STORE=models/store/library.parquet SEED=<track_id>

# 4. export the encoder to ONNX and quantize
make export
make quantize
make verify
make bench ONNX=models/onnx/mert_v1_95m_5s.int8.onnx
```

Run the FMA-small genre-recall eval (heavy first time — downloads ~7.5 GB):

```bash
make download-fma
make embed ROOT=data/raw/fma_small OUT=models/store/fma.parquet
make eval STORE=models/store/fma.parquet DATASET=fma_small
```

Random-baseline recall@10 on FMA-small's 8 balanced genres is `0.125`. The v0 quality gate is `>= 0.65`.

## v0.5: CLAP-distilled student

The shipping direction is a small CNN over 5-second log-mel windows, distilled
from LAION-CLAP music embeddings. The teacher stays in this research repo; the
student checkpoint/ONNX is the candidate on-device artifact.

```bash
make install-train
make download-fma
make download-clap-music

# Cache LAION-CLAP music teacher embeddings for FMA-small.
make distill-cache WINDOWS=1 BATCH=8 WANDB_PROJECT=latentjam-research

# Train the 5-15M parameter mel-CNN student and log curves to W&B.
make distill-train EPOCHS=5 BATCH=32 WANDB_PROJECT=latentjam-research

# Compare student vs cached LAION-CLAP teacher on FMA test rows.
make student-benchmark SPLIT=test WANDB_PROJECT=latentjam-research
# Include teacher latency too when you have the CLAP checkpoint installed.
make student-benchmark SPLIT=test TEACHER_LATENCY=1 WANDB_PROJECT=latentjam-research

# Export the student CNN; Android must reproduce the same log-mel frontend.
make student-export
make student-quantize
```

The benchmark logs student-vs-teacher cosine, nearest-neighbor top-k overlap,
FMA genre recall for both embeddings, parameter count, and local CPU latency.
With `TEACHER_LATENCY=1`, it also times LAION-CLAP on the same 5-second clip.
For quick wiring checks, add `LIMIT=...` or `LIMIT_TRAIN=... LIMIT_VAL=...`.

## Experiment tracking

`run_eval`, `conversion.verify`, `conversion.bench`, and the student CLIs can stream their JSON reports to Weights & Biases. wandb is opt-in: pass `--wandb-project latentjam-research` (and optionally `--wandb-tag ...`) on any of these CLIs. Without that flag, no wandb code runs and no network calls happen.

```bash
wandb login                       # one-time
make eval STORE=models/store/fma.parquet DATASET=fma_small \
    -- --wandb-project latentjam-research --wandb-tag dataset:fma --wandb-tag baseline
```

See `CLAUDE.md` for the full flag list and the `model_version` pinning convention used in `wandb.config`.

## Layout

```text
latentjam-research/
├── CLAUDE.md             # conventions, pinned recipe, model_version naming
├── Makefile              # install, embed, export, quantize, verify, eval, bench
├── pyproject.toml        # deps + ruff/pytest/mypy config
├── src/
│   ├── encoder/          # audio_io, MertEncoder, batching, content fingerprint
│   ├── predictor/        # EmbeddingStore (parquet), KNN, CLI, mutagen tags
│   ├── eval/             # FMA + MagnaTagATune loaders, metrics, run_eval
│   └── conversion/       # ONNX export, INT8 quantization, verify, bench
├── scripts/              # one-off CLIs (downloads)
├── tests/                # unit tests; slow tests opt-in via `-m slow`
├── notebooks/            # 00_encoder_smoke.ipynb, 01_fma_recall.ipynb
├── data/                 # gitignored — raw datasets and user audio
└── models/               # gitignored — HF cache, parquet stores, ONNX artifacts
```

## License

This repo is MIT. The MERT-v1-95M weights it loads from HuggingFace are CC-BY-NC-4.0 and **cannot ship inside the GPL-3 Android app** — see `CLAUDE.md` for the planned distillation path to a permissive student.
