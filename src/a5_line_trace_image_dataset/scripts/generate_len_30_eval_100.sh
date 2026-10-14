#!/usr/bin/env bash
# Generate 100 unpacked eval samples, each with exactly 30 permutations.
set -euo pipefail

export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/data/a5_line_trace_image/len_30_500k}"
# This matches the val split stream from a joint train,val generation run.
EVAL_SEED="${EVAL_SEED:-1000003}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" src/a5_line_trace_image_dataset/generate_dataset.py \
  --splits val \
  --num_samples_per_split 100 \
  --patch_size 16 \
  --image_width 976 \
  --column_stride 2 \
  --num_permutations 30 \
  --line_width 4 \
  --horizontal_line_width 3 \
  --displacement3_line_width 3 \
  --node_diameter 9 \
  --output_dir "${OUT_DIR}" \
  --seed "${EVAL_SEED}"

"${PYTHON_BIN}" src/a5_line_trace_image_dataset/make_inspection_html.py \
  --metadata_csv "${OUT_DIR}/val/metadata.csv" \
  --output_html "${OUT_DIR}/val/inspection.html" \
  --title "A5 line-tracing eval set: fixed length 30 (100 samples)"

echo "Eval data:       ${OUT_DIR}/val"
echo "Inspection page: ${OUT_DIR}/val/inspection.html"
