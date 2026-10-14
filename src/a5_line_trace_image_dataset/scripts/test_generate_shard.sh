#!/usr/bin/env bash
set -euo pipefail

# Small shard-mode smoke test. Run from project root.
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(pwd)"
OUT_DIR="${PROJECT_ROOT}/data/a5_line_trace_image/debug_shard"

python "${PROJECT_ROOT}/src/a5_line_trace_image_dataset/generate_dataset.py" \
  --splits test \
  --num_samples_per_split 16 \
  --patch_size 16 \
  --image_width 1024 \
  --num_permutations auto \
  --column_stride 4 \
  --line_width 4 \
  --horizontal_line_width 3 \
  --displacement3_line_width 3 \
  --node_diameter 9 \
  --pack \
  --shard_size 8 \
  --output_dir "${OUT_DIR}" \
  --seed 0

python "${PROJECT_ROOT}/src/a5_line_trace_image_dataset/eval.py" \
  --metadata_csv "${OUT_DIR}/test/metadata.csv" \
  --data_root "${OUT_DIR}/test" \
  --pred_root "${OUT_DIR}/test" \
  --output_json "${OUT_DIR}/test/eval_gt.json" \
  --output_jsonl "${OUT_DIR}/test/eval_gt.jsonl"
