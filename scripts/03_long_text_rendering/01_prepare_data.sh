#!/usr/bin/env bash
# 01_prepare_data.sh — generate the TinyStories no-edit text render dataset and
# freeze a provenance manifest.
#
# train = first 300k TinyStories train rows (PNG, no edit image); the HF
# validation split becomes `test` (validation=test). The paper's 100 held-out
# test is taken as the FIRST 100 rows of test/metadata.csv by the eval stages
# (--eval_max_samples 100) — the original 100-test manifest was not recovered
# Images are rendered white-bg/black-text via src/text_dataset.
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
DATASET="$(CFG data.dataset_name roneneldan/TinyStories)"
SPLITS="$(CFG data.splits train,validation)"
MAPPING="$(CFG data.split_mapping train=train,validation=test)"
MAX_SAMPLES="$(CFG data.max_samples 300000)"
WORKERS="$(CFG data.num_workers 8)"
NO_EDIT="$(CFG data.no_edit_image true)"
IMG_W="$(CFG data.image_width 384)"
IMG_H="$(CFG data.image_height 384)"
PADDING="$(CFG data.padding 16)"
FONT_NAME="$(CFG data.font_name Consolas)"
FONT_SIZE="$(CFG data.font_size 13)"
LINE_SPACING="$(CFG data.line_spacing 4)"
PARA_GAP="$(CFG data.paragraph_gap_lines 1)"
SHUFFLE="$(CFG data.shuffle false)"

# image_text import contract: text_dataset/* uses plain `from utils import ...` when
# launched by absolute path, and its `datasets` import needs the /src root.
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "$DATA_ROOT"
python "$COMMON/freeze_generated_data.py" --root "$DATA_ROOT" --check_destination_only

EXTRA=()
[[ "$SHUFFLE" == "true" ]] && EXTRA+=(--shuffle)
[[ "$NO_EDIT" == "true" ]] && EXTRA+=(--no_edit_image)

echo "[01_prepare_data] generating TinyStories no-edit renders -> $DATA_ROOT"
python "$REPO_ROOT/src/text_dataset/generate_tinystories.py" \
  --dataset_name "$DATASET" \
  --splits "$SPLITS" \
  --split_mapping "$MAPPING" \
  --max_samples "$MAX_SAMPLES" \
  --num_workers "$WORKERS" \
  "${EXTRA[@]}" \
  --image_width "$IMG_W" \
  --image_height "$IMG_H" \
  --padding "$PADDING" \
  --font_name "$FONT_NAME" \
  --font_size "$FONT_SIZE" \
  --line_spacing "$LINE_SPACING" \
  --paragraph_gap_lines "$PARA_GAP" \
  --hf_endpoint "${HF_ENDPOINT:-https://hf-mirror.com}" \
  --output_dir "$DATA_ROOT"

python "$COMMON/freeze_generated_data.py" --config "$CONFIG" --root "$DATA_ROOT"
echo "[01_prepare_data] done -> $DATA_ROOT"
