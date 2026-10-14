#!/usr/bin/env bash
# 01_prepare_data.sh — generate + freeze the 200-instance m=50 nfe_eval val split.
#
# Reuses the main A5 line-tracing generator with 01's geometry (stride 4 /
# patch 16 / 80x3216, m=50) and a generation seed distinct from 01's (config
# data.seed), producing exactly the `val` split under config data.data_root.
# This is the D04 held-out set — NOT the D01 20-sample main metric. A manifest
# captures seed/split/geometry + metadata digest; the recovered 200-instance
# source ID manifest is not fully recovered (see the config data.seed comment).
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
  exec python "$COMMON/provision_nfe_data.py" --config "$CONFIG"
fi
[[ "$SOURCE_MODE" == "regenerate" ]] || { echo "Unknown data.source_mode: $SOURCE_MODE (expected hf|inherit_original|regenerate)" >&2; exit 1; }

DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
NUM_EVAL="$(CFG data.num_eval_samples 200)"
M="$(CFG data.num_permutations 50)"
PATCH="$(CFG data.patch_size 16)"
WIDTH="$(CFG data.image_width 3216)"
STRIDE="$(CFG data.column_stride 4)"
LW="$(CFG data.line_width 4)"
HLW="$(CFG data.horizontal_line_width 3)"
DLW="$(CFG data.displacement3_line_width 3)"
ND="$(CFG data.node_diameter 9)"
SEED="$(CFG data.seed 1000)"
PACK="$(CFG data.pack true)"
SHARD_SIZE="$(CFG data.shard_size 10000)"
EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "$DATA_ROOT"
python "$COMMON/freeze_generated_data.py" --root "$DATA_ROOT" --check_destination_only
python "$REPO_ROOT/src/a5_line_trace_image_dataset/generate_dataset.py" \
  --splits val \
  --num_samples_per_split "$NUM_EVAL" \
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

# Manifest for provenance (seed/split/frozen geometry + metadata digest).
python "$COMMON/freeze_generated_data.py" --config "$CONFIG" --root "$DATA_ROOT"
echo "[01_prepare_data] done -> $DATA_ROOT"
