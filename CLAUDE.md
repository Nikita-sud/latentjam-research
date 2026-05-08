# latentjam-research — agent guide

Research workspace for the on-device ML recommender of **latentjam** (privacy-first Android music player, forked from Auxio, GPL-3). The Android app lives in the sibling repo `~/Documents/LJ/latentjam/`.

## Repo split

- **This repo** is laptop-only: build embedding stores, train/eval models, export ONNX artifacts.
- **App repo** consumes the artifacts at integration time (out of scope here).
- The app cannot import from this repo. Everything that crosses the boundary does so as a built artifact (e.g. `*.onnx`, parquet/npy embedding stores, sqlite DBs).

## Layout

| Path | Role |
|---|---|
| `src/encoder/` | Audio loading + the `MertEncoder` wrapper. Library code, importable. |
| `src/predictor/` | Embedding store + KNN + CLI. v0 has no learned model. |
| `src/eval/` | Dataset loaders (FMA, MagnaTagATune) + content-only metrics. |
| `src/conversion/` | PyTorch → ONNX export, INT8 quantization, verification, benchmarking. |
| `scripts/` | One-off CLIs with side effects (downloads, profiling). Not a package. |
| `tests/` | Mirrors `src/`. Network/large-file tests marked `@pytest.mark.slow`. |
| `notebooks/` | Exploratory Jupyter. Not source of truth. |
| `data/` | Gitignored. Raw datasets and user audio. |
| `models/` | Gitignored. HF cache, embedding stores, ONNX artifacts. |

## Conventions

- **`src/` is library code** (pure, importable, no side effects on import). **`scripts/` is one-off CLIs** (network, hardcoded paths, dataset URLs OK).
- **Tests touching network or large files**: mark `@pytest.mark.slow`. They're skipped by default (`addopts` in `pyproject.toml`); run with `pytest -m slow` explicitly.
- **CLI args use absolute paths**, never relative.
- **No emoji in source files.**
- **No comments that explain WHAT** the code does — names should do that. Comments only for non-obvious WHY (a constraint, an invariant, a workaround).

## Pinned recipe (do not silently change)

The embedding pipeline has a fixed shape so that ONNX export and on-device inference stay deterministic:

- **Audio**: mono float32 in `[-1, 1]`, resampled to **24 kHz** via `resampy`.
- **Window**: fixed **5 s = 120,000 samples**, **50 % overlap** between windows when a track is longer.
- **Encoder**: `m-a-p/MERT-v1-95M` from HuggingFace, `output_hidden_states=True`.
- **Pooling**: mean-pool layers `[5, 6, 7, 8]` → mean-pool over time → L2-normalize → 768-d vector per window. Track embedding = mean of window embeddings, re-normalized.
- **`model_version` string**: `mert-v1-95m@layers5-8/meanpool/v1`. Bumping any pooling/normalization detail bumps the suffix. Stored on every embedding row so stale cache entries are detectable.

If you change any of the above, the change must propagate to: `mert_encoder.py`, the ONNX export wrapper in `conversion/export_onnx.py`, the verification thresholds in `conversion/verify.py`, and the version suffix.

## Embedding store schema (`tracks.parquet`)

| Column | Type | Notes |
|---|---|---|
| `track_id` | string | xxhash64 of `(file_size, first 1 MiB, last 1 MiB)`. Stable across path moves. |
| `path` | string | Absolute path at ingest time (informational; may go stale). |
| `title`, `artist`, `album`, `genre` | string \| null | Best-effort metadata from tags / dataset. |
| `year` | int32 \| null | |
| `model_version` | string | E.g. `mert-v1-95m@layers5-8/meanpool/v1`. |
| `embedding` | list&lt;float32&gt; | L2-normalized; length matches `embedding_dim`. |
| `embedding_dim` | int32 | 768 for v0. |

Optional sidecar `embeddings.npy` is a `(N, D)` float32 matrix row-aligned to the parquet, regenerated on `EmbeddingStore.save()`.

## License caveats

- MERT-v1-95M weights are **CC-BY-NC-4.0**. Use in this repo only. Do **not** ship the weights — and arguably not the embeddings — inside the GPL-3 app. The path to a shipping model is distillation into a permissive student (PaSST-S / PANNs CNN14, both Apache-2.0). That work is v0.5+, out of v0 scope.
- This repo is MIT.

## Mobile runtime

Target is **`onnxruntime-android` full AAR**, NOT the reduced "ORT Mobile" build. The reduced build silently strips operators that transformer audio models need. This repo only emits ONNX artifacts; the AAR choice belongs to the app repo at integration time.

## Common commands

```bash
make install                    # pip install -e ".[dev]"
make lint                       # ruff check .
make test                       # pytest (skips slow)
make test-slow                  # pytest -m slow

make embed ROOT=<dir> OUT=<store.parquet>
make export                     # → models/onnx/mert_v1_95m_5s.onnx
make quantize                   # → models/onnx/mert_v1_95m_5s.int8.onnx
make eval STORE=... DATASET=fma_small|mtat
make bench ONNX=...
```
