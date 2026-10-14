#!/usr/bin/env bash
# Generate the A5 line-tracing dataset with uniformly random length in [1, 50],
# with an ADAPTIVE canvas: each sample's image width is sized to exactly fit its
# trace (active_columns * patch_size), so there is no trailing white space.
#
#   train:                300,000 samples  (length uniform 1..50, variable width)
#   val / test_random:      100 samples each (length uniform 1..50, variable width)
#   test_depth:             100 samples      (fixed depth sweep 5,10,20,30,40,50)
#
# --image_width=1616 is the MAX cap (canvas_columns=101, column_stride=2 ->
# max_steps=50). With --adaptive_canvas the per-sample width is smaller for
# shorter traces: length 1 -> 48px, length 50 -> 1616px.
# Shard (pack) mode, compatible with src/image_text/data.py and src/image_only/data.py.
# Variable per-sample widths => train with batch_size=1 (no torch.compile).
#
# Run from the project root, or via tmux (see the package README).
set -eo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate the project conda env (has Pillow, numpy, tqdm) — the same env used
# for training (historical swap-eval runbook; not bundled in the repo). Point CONDA_SH at
# your own conda.sh if `conda` is not already on PATH. `set -u` stays off:
# conda's deactivate hook trips on unbound variables.
[[ -n "${CONDA_SH:-}" ]] && source "${CONDA_SH}"
conda activate "${CONDA_ENV:-image2}"

OUT_DIR="data/a5_line_trace_image/len_1_50_shard"

echo "============================================================"
echo " A5 line-trace dataset generation"
echo "   length: 1-50 uniform random (num_permutations_range)"
echo "   canvas: ADAPTIVE (per-sample width = active_columns * patch_size)"
echo "   cap: image_width=1616 (101 cols), column_stride=2, max_steps=50"
echo "   splits: train=300000, val/test_random/test_depth=100 each"
echo "   test_depth sweep: 5,10,20,30,40,50"
echo "   mode: pack (shard), shard_size=10000"
echo "   output: ${OUT_DIR}"
echo "   start: $(date)"
echo "============================================================"

python src/a5_line_trace_image_dataset/generate_dataset.py \
  --splits train,val,test_random,test_depth \
  --num_samples_per_split 100 \
  --num_train_samples 300000 \
  --patch_size 16 \
  --image_width 1616 \
  --column_stride 2 \
  --num_permutations_range 1,50 \
  --adaptive_canvas \
  --depth_values 5,10,20,30,40,50 \
  --line_width 4 \
  --horizontal_line_width 3 \
  --displacement3_line_width 3 \
  --node_diameter 9 \
  --pack \
  --shard_size 10000 \
  --output_dir "${OUT_DIR}" \
  --seed 0

echo
echo "============================================================"
echo " GENERATION COMPLETE"
echo "   end:   $(date)"
echo "   output: ${PROJECT_ROOT}/${OUT_DIR}"
echo "============================================================"
find "${OUT_DIR}" -maxdepth 2 -type d | sort
