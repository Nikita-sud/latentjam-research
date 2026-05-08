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
