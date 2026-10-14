#!/usr/bin/env bash
# 05_diffthinker_controlled/<task>/run.sh — orchestrate the seven stages (+ examples).
#
#   run.sh [--profile paper|smoke] [--dry-run] [--stage 03_train_vanilla] [stage names...]
#
# With --dry-run it resolves the config and prints its path plus stage names. With
# one or more stage names it runs just those (in argument order); otherwise it
# runs the full sequence. Stage scripts are independently callable.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_d="$SELF_DIR"; while [[ "$_d" != "${_d%/*}" ]]; do
  [[ -f "$_d/scripts/common/env.sh" ]] && { COMMON="$_d/scripts/common"; break; }
  _d="${_d%/*}"
done
# shellcheck disable=SC1090
source "$COMMON/env.sh"

CONFIG="${CONFIG_OVERRIDE:-$SELF_DIR/config.yaml}"
export CONFIG_OVERRIDE="$CONFIG"
PROFILE="paper"
DRY_RUN=0
STAGES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --stage) STAGES+=("$2"); shift 2 ;;
    *) STAGES+=("$1"); shift ;;
  esac
done

STAGE_ORDER=(01_prepare_data 02_measure_flops 03_train_vanilla 04_train_selfcond
             05_match_checkpoints 06_evaluate 07_summarize)
if [[ ${#STAGES[@]} -eq 0 ]]; then STAGES=("${STAGE_ORDER[@]}"); fi

# Preflight on the resolved config (offline; no model/API/GPU).
"$COMMON/preflight.sh" --config "$CONFIG" --profile "$PROFILE" || exit 1

run_stage() {
  local name="$1"
  [[ -f "$SELF_DIR/$name.sh" ]] || { echo "run: missing stage $SELF_DIR/$name.sh" >&2; return 1; }
  echo "=== [$PROFILE] $name ($(date -u +%H:%M:%S)) ==="
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [dry-run] PROFILE=$PROFILE CONFIG=$CONFIG stage=$name"
    return 0
  fi
  PROFILE="$PROFILE" bash "$SELF_DIR/$name.sh"
}

rc=0
for s in "${STAGES[@]}"; do
  run_stage "$s" || { rc=1; echo "run: stage $s failed" >&2; break; }
done

echo "=== $(basename "$SELF_DIR") run finished (rc=$rc) ==="
exit $rc