#!/usr/bin/env bash
set -euo pipefail

# Full/default shard generation. Run from project root.
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(pwd)"
OUT_DIR="${PROJECT_ROOT}/data/a5_line_trace_image/full_data_shard_xl"

python "${PROJECT_ROOT}/src/a5_line_trace_image_dataset/generate_dataset.py" \
  --splits train,val,test_random,test_depth \
  --num_samples_per_split 10000 \
  --num_train_samples 1000000 \
  --patch_size 16 \
  --image_width 1024 \
  --num_permutations auto \
  --column_stride 4 \
  --depth_values 4,8,12,15 \
  --line_width 4 \
  --horizontal_line_width 3 \
  --displacement3_line_width 3 \
  --node_diameter 9 \
  --pack \
  --shard_size 10000 \
  --output_dir "${OUT_DIR}" \
  --seed 0
