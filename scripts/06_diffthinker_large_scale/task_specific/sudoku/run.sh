#!/usr/bin/env bash
# 06_diffthinker_large_scale/task_specific/vsp/run.sh — orchestrate the six stages.
#
#   run.sh [--profile paper|smoke] [--dry-run] [--stage NAME] [stage names...]
#
# Stages: 01_prepare_data 02_measure_flops 03_train_selfcond 04_evaluate 05_score 06_summarize
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

STAGE_ORDER=(01_prepare_data 02_measure_flops 03_train_selfcond 04_evaluate
             05_score 06_summarize)
if [[ ${#STAGES[@]} -eq 0 ]]; then STAGES=("${STAGE_ORDER[@]}"); fi

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

echo "=== 06_diffthinker_large_scale run finished (rc=$rc) ==="
exit $rc