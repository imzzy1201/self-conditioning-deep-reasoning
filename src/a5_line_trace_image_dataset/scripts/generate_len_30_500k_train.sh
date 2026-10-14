#!/usr/bin/env bash
# Generate 500,000 packed training samples, each with exactly 30 permutations.
set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/a5_line_trace_image/len_30_500k}"
TRAIN_SEED="${TRAIN_SEED:-0}"
SHARD_SIZE="${SHARD_SIZE:-10000}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" src/a5_line_trace_image_dataset/generate_dataset.py \
  --splits train \
  --num_train_samples 500000 \
  --patch_size 16 \
  --image_width 976 \
  --column_stride 2 \
  --num_permutations 30 \
  --line_width 4 \
  --horizontal_line_width 3 \
  --displacement3_line_width 3 \
  --node_diameter 9 \
  --pack \
  --shard_size "${SHARD_SIZE}" \
  --output_dir "${OUT_DIR}" \
  --seed "${TRAIN_SEED}"

echo "Training data: ${OUT_DIR}/train"
