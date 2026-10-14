#!/usr/bin/env bash
# 05_match_checkpoints.sh — apply measured FLOPs rates, audit, then select the
# protocol-compliant pair (docs/flops_protocol.md).
#
#   apply  : write cumulative_flops into every checkpoint of both runs (per-method
#            rate from <outputs.flops_dir>/<method>/flops.json — never shared).
#   audit  : hard-fail if any checkpoint lacks cumulative_flops.
#   match  : anchor at paper_selfcond_step; pair the highest SelfCond step to
#            the nearest Vanilla with F_vanilla >= F_selfcond (no gap threshold).
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

PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
AUDIT_MANIFEST="$(resolve_path "$(CFG outputs.audit_manifest)")"
PAPER_SC="$(CFG checkpoints.paper_selfcond_step 16000)"

FLOPSCLI="python $REPO_ROOT/src/experiment_tools/flops_cli.py"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/experiment_tools:${PYTHONPATH:-}"

echo "[05_match_checkpoints] apply"
$FLOPSCLI apply --config "$CONFIG"

echo "[05_match_checkpoints] audit"
$FLOPSCLI audit --config "$CONFIG" --out "$AUDIT_MANIFEST"

echo "[05_match_checkpoints] match (paper_selfcond_step=$PAPER_SC)"
$FLOPSCLI match --config "$CONFIG" --out "$PAIR_MANIFEST" \
  --paper_selfcond_step "$PAPER_SC"

echo "[05_match_checkpoints] done -> $PAIR_MANIFEST"