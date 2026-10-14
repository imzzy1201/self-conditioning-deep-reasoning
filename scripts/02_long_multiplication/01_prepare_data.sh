#!/usr/bin/env bash
# 01_prepare_data.sh — generate the fixed 5-7 digit multiplication dataset and
# freeze a provenance manifest (train=200k, test=100; digits uniform in [5,7]).
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
TRAIN_DIR="$DATA_ROOT/data_200k"
TEST_DIR="$DATA_ROOT/data_test100"
MIN_DIGITS="$(CFG data.min_digits 5)"
MAX_DIGITS="$(CFG data.max_digits 7)"
TRAIN_N="$(CFG data.num_train_samples 200000)"
EVAL_N="$(CFG data.num_eval_samples 100)"
CELL="$(CFG data.cell_size 16)"
PADDING="$(CFG data.padding 16)"
FONT="$(CFG data.font_name 'DejaVu Sans Mono')"
PACK="$(CFG data.pack true)"
SHARD_SIZE="$(CFG data.shard_size 10000)"
TRAIN_SEED="$(CFG data.train_seed 42)"
EVAL_SEED="$(CFG data.eval_seed 1234)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "$DATA_ROOT"
python "$COMMON/freeze_generated_data.py" --root "$DATA_ROOT" --check_destination_only

run_gen() {
  local out="$1" n="$2" seed="$3"
  mkdir -p "$out"
  python "$REPO_ROOT/src/long_multiplication_dataset/generate_random_digits_dataset.py" \
    --min_digits "$MIN_DIGITS" --max_digits "$MAX_DIGITS" \
    --num_samples "$n" --output_dir "$out" \
    --cell_size "$CELL" --padding "$PADDING" --font_name "$FONT" \
    $([[ "$PACK" == "true" ]] && echo --pack) --shard_size "$SHARD_SIZE" \
    --seed "$seed"
}

run_gen "$TRAIN_DIR" "$TRAIN_N" "$TRAIN_SEED"
run_gen "$TEST_DIR" "$EVAL_N" "$EVAL_SEED"

python "$COMMON/freeze_generated_data.py" --config "$CONFIG" --root "$DATA_ROOT"
echo "[01_prepare_data] done -> $DATA_ROOT"
