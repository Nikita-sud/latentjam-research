#!/usr/bin/env bash
# End-to-end SSL training bootstrap for a fresh RunPod pod.
#
# Designed to be runnable on any "PyTorch 2.x + CUDA 12" RunPod template.
# Idempotent: each phase checks whether its output already exists, so re-runs
# (or pod restarts) only redo missing work.
#
# Usage on the pod (after SSH or via web terminal):
#
#   export WANDB_API_KEY="..."          # optional but recommended
#   export SUBSET=medium                # small | medium | large
#   export EPOCHS=30
#   export BATCH_SIZE=512
#   bash scripts/runpod_train.sh
#
# Or one-shot (clone + run):
#
#   curl -fsSL https://raw.githubusercontent.com/Nikita-sud/latentjam-research/main/scripts/runpod_train.sh \
#     | WANDB_API_KEY=... SUBSET=medium bash
#
# RunPod tip: keep state in /workspace (persistent volume); /root and /tmp are
# ephemeral and get wiped on pod restart. This script defaults WORK_DIR there.

set -euo pipefail

# ---------- Config (env-overridable) ----------
REPO_URL="${REPO_URL:-https://github.com/Nikita-sud/latentjam-research}"
WORK_DIR="${WORK_DIR:-/workspace/latentjam-research}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/raw}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache/ssl_audio}"

SUBSET="${SUBSET:-medium}"          # small | medium | large
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1.5e-3}"
DECODE_WORKERS="${DECODE_WORKERS:-12}"
PROJECTOR_DIM="${PROJECTOR_DIM:-2048}"
GAIN_JITTER_DB="${GAIN_JITTER_DB:-4.0}"
POLARITY_FLIP_PROB="${POLARITY_FLIP_PROB:-0.0}"
ARTIST_POSITIVE_PROB="${ARTIST_POSITIVE_PROB:-0.0}"

WANDB_PROJECT="${WANDB_PROJECT:-latentjam-research}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ssl-vicreg-fma-${SUBSET}-runpod-bs${BATCH_SIZE}-ep${EPOCHS}}"

# Sync target (optional). If set, uploads the best checkpoint here at the end.
# Examples: "remote:checkpoints/" for rclone, or any rsync URL.
SYNC_TARGET="${SYNC_TARGET:-}"

CHECKPOINT_PATH="${CHECKPOINT_DIR}/mel_cnn_ssl_${SUBSET}_runpod.pt"

# ---------- Pretty banner ----------
echo "==============================================="
echo "latentjam-research — SSL training on RunPod"
echo "==============================================="
echo "  subset      : ${SUBSET}"
echo "  epochs      : ${EPOCHS}"
echo "  batch size  : ${BATCH_SIZE}"
echo "  lr          : ${LR}"
echo "  projector   : ${PROJECTOR_DIM}-d"
echo "  work dir    : ${WORK_DIR}"
echo "  data root   : ${DATA_ROOT}"
echo "  ckpt dir    : ${CHECKPOINT_DIR}"
echo "  wandb run   : ${WANDB_RUN_NAME}"
echo "==============================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || \
    echo "(no nvidia-smi; running CPU/MPS path)"
echo ""

mkdir -p "$WORK_DIR" "$DATA_ROOT" "$CHECKPOINT_DIR" "$CACHE_DIR"

# ---------- Phase 1: Repo ----------
if [ ! -d "$WORK_DIR/.git" ]; then
    echo "[phase 1/5] cloning repo into $WORK_DIR"
    git clone "$REPO_URL" "$WORK_DIR"
else
    echo "[phase 1/5] repo present; pulling latest"
    git -C "$WORK_DIR" pull --rebase --autostash
fi
cd "$WORK_DIR"

# Symlink data + cache + checkpoint dirs into the repo so the existing
# script defaults work without -- flag overrides.
mkdir -p "$WORK_DIR/data"
[ -L "$WORK_DIR/data/raw" ] || rm -rf "$WORK_DIR/data/raw" 2>/dev/null || true
ln -sfn "$DATA_ROOT" "$WORK_DIR/data/raw"

# ---------- Phase 2: Python deps ----------
echo "[phase 2/5] python deps"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip > /dev/null
# Install with the [dev,train] extras so wandb + clap-related deps are present.
pip install --quiet -e ".[dev,train]"

# ---------- Phase 3: Data ----------
SUBSET_DIR="${DATA_ROOT}/fma_${SUBSET}"
METADATA_DIR="${DATA_ROOT}/fma_metadata"

if [ -d "$SUBSET_DIR" ] && [ -n "$(ls -A "$SUBSET_DIR" 2>/dev/null)" ] && [ -d "$METADATA_DIR" ]; then
    echo "[phase 3/5] FMA-${SUBSET} already present at ${SUBSET_DIR}; skipping download"
else
    echo "[phase 3/5] downloading FMA-${SUBSET}"
    python scripts/download_fma.py --subset "$SUBSET" --out "$DATA_ROOT"
fi

# ---------- Phase 4: W&B auth ----------
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[phase 4/5] wandb login"
    wandb login --relogin "$WANDB_API_KEY" > /dev/null 2>&1 || true
    WANDB_FLAGS=(--wandb-project "$WANDB_PROJECT" --wandb-run-name "$WANDB_RUN_NAME"
        --wandb-tag "method:vicreg"
        --wandb-tag "dataset:fma-${SUBSET}"
        --wandb-tag "device:cuda"
        --wandb-tag "host:runpod"
        --wandb-tag "bs:${BATCH_SIZE}")
else
    echo "[phase 4/5] WANDB_API_KEY not set; running without wandb"
    WANDB_FLAGS=()
fi

# ---------- Phase 5: Train ----------
echo "[phase 5/5] training"

# Threading hints for predictable performance on multi-GPU pods.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
# Cache HF / wandb artifacts on the persistent volume rather than ephemeral
# /root so they survive pod restarts.
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface}"
export WANDB_DIR="${WANDB_DIR:-/workspace/.cache/wandb}"
mkdir -p "$HF_HOME" "$WANDB_DIR"

START_TS=$(date +%s)

python -m student.train_ssl \
    --audio-root "$SUBSET_DIR" \
    --metadata-root "$METADATA_DIR" \
    --subset "$SUBSET" \
    --device auto \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --decode-workers "$DECODE_WORKERS" \
    --cache-dir "$CACHE_DIR" \
    --projector-dim "$PROJECTOR_DIM" \
    --gain-jitter-db "$GAIN_JITTER_DB" \
    --polarity-flip-prob "$POLARITY_FLIP_PROB" \
    --artist-positive-prob "$ARTIST_POSITIVE_PROB" \
    --out "$CHECKPOINT_PATH" \
    "${WANDB_FLAGS[@]}"

END_TS=$(date +%s)
DURATION_MIN=$(( (END_TS - START_TS) / 60 ))
echo ""
echo "training finished in ${DURATION_MIN} minutes"
echo "checkpoint: $CHECKPOINT_PATH"

# ---------- Optional: sync checkpoint elsewhere ----------
if [ -n "$SYNC_TARGET" ]; then
    if command -v rclone > /dev/null && [[ "$SYNC_TARGET" == *:* ]]; then
        echo "[sync] rclone copy $CHECKPOINT_PATH -> $SYNC_TARGET"
        rclone copy "$CHECKPOINT_PATH" "$SYNC_TARGET"
    elif command -v rsync > /dev/null; then
        echo "[sync] rsync $CHECKPOINT_PATH -> $SYNC_TARGET"
        rsync -avh "$CHECKPOINT_PATH" "$SYNC_TARGET"
    else
        echo "[sync] SYNC_TARGET set but neither rclone nor rsync are available"
    fi
fi

echo ""
echo "==============================================="
echo "Done. Pod can be stopped/terminated when ready."
echo "==============================================="
