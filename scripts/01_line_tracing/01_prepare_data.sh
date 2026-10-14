#!/usr/bin/env bash
# 01_prepare_data.sh — generate the fixed m=50 line-tracing dataset and freeze a manifest.
#
# Train=1,000,000 samples (paper), val=20 held-out (paper main eval). GT sanity
# (LCP==50 on references, 0 on a corrupted first stage) is asserted in
# 06_evaluate.sh's geometry check rather than here.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"
CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"
CFG() { python "$COMMON/read_config.py" "$CONFIG" "$1" "${2:-}"; }

SOURCE_MODE="${DATA_SOURCE_MODE:-$(CFG data.source_mode hf)}"
if [[ "$SOURCE_MODE" == "hf" ]]; then
  exec python "$COMMON/provision_hf_data.py" --config "$CONFIG"
fi
if [[ "$SOURCE_MODE" == "inherit_original" ]]; then
  exec python "$COMMON/provision_original_data.py" --config "$CONFIG"
fi
[[ "$SOURCE_MODE" == "regenerate" ]] || { echo "Unknown data.source_mode: $SOURCE_MODE (expected hf|inherit_original|regenerate)" >&2; exit 1; }

DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
NUM_TRAIN="$(CFG data.num_train_samples 1000000)"
NUM_EVAL="$(CFG data.num_eval_samples 20)"
M="$(CFG data.num_permutations 50)"
PATCH="$(CFG data.patch_size 16)"
WIDTH="$(CFG data.image_width 3216)"
STRIDE="$(CFG data.column_stride 4)"
LW="$(CFG data.line_width 4)"
HLW="$(CFG data.horizontal_line_width 3)"
DLW="$(CFG data.displacement3_line_width 3)"
ND="$(CFG data.node_diameter 9)"
SEED="$(CFG data.seed 0)"
PACK="$(CFG data.pack true)"
SHARD_SIZE="$(CFG data.shard_size 10000)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "$DATA_ROOT"
python "$COMMON/freeze_generated_data.py" --root "$DATA_ROOT" --check_destination_only
python "$REPO_ROOT/src/a5_line_trace_image_dataset/generate_dataset.py" \
  --splits train,val \
  --num_samples_per_split "$NUM_EVAL" \
  --num_train_samples "$NUM_TRAIN" \
  --num_permutations "$M" \
  --patch_size "$PATCH" \
  --image_width "$WIDTH" \
  --column_stride "$STRIDE" \
  --line_width "$LW" \
  --horizontal_line_width "$HLW" \
  --displacement3_line_width "$DLW" \
  --node_diameter "$ND" \
  $([[ "$PACK" == "true" ]] && echo --pack) \
  --shard_size "$SHARD_SIZE" \
  --output_dir "$DATA_ROOT" \
  --seed "$SEED"

# Manifest for provenance (seed/split/frozen geometry + metadata digests).
python "$COMMON/freeze_generated_data.py" --config "$CONFIG" --root "$DATA_ROOT"
echo "[01_prepare_data] done -> $DATA_ROOT"
