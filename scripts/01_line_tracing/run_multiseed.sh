#!/usr/bin/env bash
# run_multiseed.sh — 3-seed line-tracing sweep (fig:multi_seed_line_tracing).
#
# Data (01) and the per-micro-batch FLOPs rate (02) are seed-invariant, so they
# are produced once. Each training seed runs its own train -> pair -> evaluate ->
# summarize chain isolated from the others (separate checkpoint/output dirs and
# a per-seed config). Pairing is per-seed via the shared flops match (each seed's
# checkpoints carry its own cumulative FLOPs); the rate itself is reused because
# forward/backward FLOPs do not depend on the sampled init.
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
CFG() { python "$COMMON/read_config.py" "$CONFIG" "$1" "${2:-}"; }
PROFILE="${PROFILE:-paper}"

# Shared prerequisites (idempotent guards).
DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
FLOPS_DIR="$(resolve_path "$(CFG outputs.flops_dir)")"
[[ -f "$DATA_ROOT/manifest.json" ]] || PROFILE="$PROFILE" bash "$SELF_DIR/01_prepare_data.sh"
[[ -f "$FLOPS_DIR/vanilla/flops.json" && -f "$FLOPS_DIR/selfcond/flops.json" ]] \
  || PROFILE="$PROFILE" bash "$SELF_DIR/02_measure_flops.sh"

SEED_LIST=$(python - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(str(s) for s in cfg.get("multiseed", {}).get("seeds", [1201, 1202, 1203])))
PY
)

declare -a SEED_CONFIGS=()
for s in $SEED_LIST; do
  SEED_CONFIG="${OUTPUT_ROOT}/01_line_tracing/multiseed/${s}/config.yaml"
  mkdir -p "$(dirname "$SEED_CONFIG")"
  python - "$CONFIG" "$SEED_CONFIG" "$s" <<'PY'
import sys, yaml
base = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
out, seed = sys.argv[2], sys.argv[3]
from pathlib import Path
scope = str(Path(out).parent)
base["train"]["seed"] = int(seed)
base["checkpoints"]["vanilla_dir"] = f"{scope}/vanilla"
base["checkpoints"]["selfcond_dir"] = f"{scope}/selfcond"
# flops_dir stays shared: the measured rate is seed-invariant.
base["outputs"]["pair_manifest"] = f"{scope}/pairs.json"
base["outputs"]["audit_manifest"] = f"{scope}/audit.json"
base["outputs"]["eval_root"] = f"{scope}/eval"
base["outputs"]["summary"] = f"{scope}/summary.json"
base["outputs"]["figure"] = f"{scope}/curves.svg"
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
print(out)
PY
  echo "[run_multiseed] seed $s -> $SEED_CONFIG"
  CONFIG_OVERRIDE="$SEED_CONFIG" PROFILE="$PROFILE" bash "$SELF_DIR/run.sh" \
    --profile "$PROFILE" \
    --stage 03_train_vanilla --stage 04_train_selfcond --stage 05_match_checkpoints \
    --stage 06_evaluate --stage 07_summarize
  SEED_CONFIGS+=("$SEED_CONFIG")
done

# Aggregate across seeds: mean CTD + sample SD (ddof=1) for each method.
python - "$OUTPUT_ROOT/01_line_tracing/multiseed" "${SEED_CONFIGS[@]}" <<'PY'
import json, sys
from pathlib import Path
from statistics import mean, stdev

root = Path(sys.argv[1])
configs = [Path(c) for c in sys.argv[2:]]
rows = []
for cfg in configs:
    summary_path = root / cfg.parent.name / "summary.json"
    if not summary_path.exists():
        rows.append({"seed": cfg.parent.name, "summary": None})
        continue
    rows.append({"seed": cfg.parent.name, "summary": json.loads(summary_path.read_text(encoding="utf-8"))})

agg = {"seeds": [], "methods": {}}
for method in ("vanilla", "selfcond"):
    vals, flops = [], []
    for r in rows:
        s = r["summary"]
        if s:

            if r["seed"] not in agg["seeds"]:
                agg["seeds"].append(r["seed"])
            v = s.get("eval", {}).get(f"{method}_ctd_mean")
            if v is not None:
                vals.append(v)
            sel = (s.get("pair") or {}).get("selected")
            if sel:
                flops.append(sel.get(f"{'vanilla' if method == 'vanilla' else 'selfcond'}_flops"))
    agg["methods"][method] = {
        "n_seeds_scored": len(vals),
        "ctd_mean": mean(vals) if vals else None,
        "ctd_sd": stdev(vals) if len(vals) > 1 else 0.0,
        "matched_flops": mean(flops) if flops else None,
    }

agg["complete"] = all(cell["n_seeds_scored"] == len(configs) for cell in agg["methods"].values())
out = root / "multiseed_summary.json"
out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
print(json.dumps(agg, indent=2))
if not agg["complete"]:
    raise SystemExit("Missing seed scores: multi-seed aggregate is incomplete")
PY

echo "[run_multiseed] done -> $OUTPUT_ROOT/01_line_tracing/multiseed/multiseed_summary.json"
