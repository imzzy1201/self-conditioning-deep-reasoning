#!/usr/bin/env bash
set -euo pipefail

# Evaluate prediction images. Run from project root.
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/a5_line_trace_image/full_data_shard/val}"
METADATA_CSV="${METADATA_CSV:-${DATA_ROOT}/metadata.csv}"
PRED_DIR="${PRED_DIR:?Set PRED_DIR to a directory containing prediction PNGs}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/a5_line_trace_image/eval}"

python "${PROJECT_ROOT}/src/a5_line_trace_image_dataset/eval.py" \
  --metadata_csv "${METADATA_CSV}" \
  --data_root "${DATA_ROOT}" \
  --pred_dir "${PRED_DIR}" \
  --output_json "${OUT_DIR}/metrics.json" \
  --output_jsonl "${OUT_DIR}/samples.jsonl"
