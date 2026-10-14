#!/usr/bin/env bash
# common/preflight.sh — validate config VALUES offline before a run (no path,
# data, GPU, or network checks). Verifies: experiment.id; save_steps ==
# eval_every_steps per method (paper profile); method.name router;
# paper_{selfcond,vanilla}_step on the trainer's save grid. Smoke profile
# requires a dedicated experiment.mode=smoke config. --stage is echoed, not
# validated. Run automatically at the top of every run.sh.
#
#   preflight.sh --config <experiment config.yaml> [--stage <name>] [--profile paper|smoke]
#   preflight.sh --check-line-endings
#
# Pure offline checks (no model download, API, or GPU). Fails nonzero on a
# broken config or inconsistent save/eval interval. Does NOT run training.
#
# `--check-line-endings` runs the byte-level CR scan over the tree instead of a
# config check (`bash -n` cannot see CRLF; see common/check_line_endings.py).

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SELF_DIR}/env.sh"

CONFIG=""
STAGE=""
PROFILE="paper"
CHECK_EOL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --check-line-endings) CHECK_EOL=1; shift ;;
    *) echo "preflight: unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ "${CHECK_EOL}" -eq 1 ]]; then
  exec python "${SELF_DIR}/check_line_endings.py"
fi

[[ -n "${CONFIG}" ]] || { echo "preflight: --config is required" >&2; exit 1; }
CONFIG="$(resolve_path "${CONFIG}")"
[[ -f "${CONFIG}" ]] || { echo "preflight: config not found: ${CONFIG}" >&2; exit 1; }

python - "${CONFIG}" "${PROFILE}" "${STAGE}" <<'PY'
import sys, os
from pathlib import Path
import yaml

cfg_path, profile, stage = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
errs = []
if profile not in ("paper", "smoke"):
    errs.append(f"Unknown profile: {profile}")
if profile == "smoke" and (cfg.get("experiment") or {}).get("mode") != "smoke":
    errs.append("smoke requires a separate config with experiment.mode=smoke and explicit small data/training budgets; --profile does not reduce paper budgets")

# experiment + train sections
exp = cfg.get("experiment") or {}
if not exp.get("id"):
    errs.append("experiment.id is required")
train = cfg.get("train") or {}

# long-multiplication-style save/eval synchronisation: if the config declares
# an eval interval and a save interval, they must match for formal runs. Some
# two-method tasks use a different cadence per method (e.g. Vanilla 12500 /
# SelfCond 5000), expressed as train.selfcond_{save,eval}_steps.
def check_sync(save_val, eval_val, label):
    # Zero explicitly disables periodic evaluation (the unified source run).
    # Positive evaluation intervals still require a matching save cadence.
    if save_val is not None and eval_val is not None and int(eval_val) > 0 and str(save_val) != str(eval_val):
        errs.append(
            f"train.{label}save_steps ({save_val}) != train.{label}eval_every_steps ({eval_val}); "
            "formal runs must save on the same cadence they evaluate"
        )

if profile == "paper":
    check_sync(train.get("save_steps"), train.get("eval_every_steps"), "")
    check_sync(train.get("selfcond_save_steps"), train.get("selfcond_eval_every_steps"), "selfcond_")

# methods must be one of the two routers
method = (cfg.get("method") or {}).get("name")
if method not in (None, "vanilla", "proj_condition"):
    errs.append(f"method.name must be vanilla|proj_condition, got {method}")

# checkpoints section (required for two-method tasks)
ck = cfg.get("checkpoints") or {}
has_pair = bool(ck.get("vanilla_dir")) and bool(ck.get("selfcond_dir"))
if has_pair and ck.get("paper_selfcond_step") is None:
    errs.append("checkpoints.paper_selfcond_step is required when both dirs are set")

# A declared paper anchor is only reproducible if the trainer's save grid lands on
# it: train.py writes periodic checkpoints on `global_step % save_steps == 0` and
# eval_every_steps never saves, so a step off the save grid is never written.
# 'final' comes from --save_final_checkpoint and needs no grid. A leaf that
# declares neither an anchor nor a cadence (reuse / eval-only) has nothing to
# prove, which is why no cadence is demanded unconditionally -- the anchor check
# below carries that requirement instead, and only for leaves that declare one.
def check_anchor(anchor, save_val, cadence, label):
    if profile != "paper" or anchor is None:
        return
    if not isinstance(anchor, int) or isinstance(anchor, bool):
        if anchor != "final":
            errs.append(f"checkpoints.{label} must be an integer step or 'final', got {anchor!r}")
        return
    if save_val in (None, 0):
        errs.append(
            f"checkpoints.{label} ({anchor}) is declared but train.{cadence} is not set; "
            "the trainer has no save grid that could produce it"
        )
    elif anchor % int(save_val) != 0:
        errs.append(
            f"checkpoints.{label} ({anchor}) is not a multiple of train.{cadence} "
            f"({save_val}); the trainer would never write that checkpoint"
        )

selfcond_cadence = "selfcond_save_steps" if train.get("selfcond_save_steps") is not None else "save_steps"
check_anchor(ck.get("paper_selfcond_step"),
             train.get(selfcond_cadence), selfcond_cadence, "paper_selfcond_step")
check_anchor(ck.get("paper_vanilla_step"),
             train.get("save_steps"), "save_steps", "paper_vanilla_step")

if errs:
    for e in errs:
        print(f"preflight: {e}", file=sys.stderr)
    sys.exit(1)

print(f"preflight OK: experiment={exp.get('id')} profile={profile} stage={stage or '(all)'}")
PY
