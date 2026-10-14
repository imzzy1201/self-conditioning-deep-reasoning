#!/usr/bin/env bash
# run_multiseed.sh — 3-seed long-text-rendering sweep (EXTENSION; the paper reports
# NO multi-seed numbers for this task, and this sweep freezes no paper value).
#
# Data (01) and the per-micro-batch rate (02) are seed-invariant and produced
# once. Each training seed runs its own train -> pair -> evaluate -> summarize
# chain in isolated dirs, anchored at SelfCond@74000 (the main-protocol anchor
# reused here, not a separate budget). Pairing is per-seed via the shared flops
# match. Aggregated cer_mean is the mean over seed-wise valid-only CER means.
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
[[ "$(CFG multiseed.enabled false)" == "true" ]] || { echo "The paper has no long-text multi-seed experiment; set multiseed.enabled=true for an additional experiment." >&2; exit 1; }
ANCHOR_SC="$(CFG multiseed.paper_selfcond_step 74000)"
ANCHOR_V="$(CFG multiseed.paper_vanilla_step 190000)"

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
  SEED_CONFIG="${OUTPUT_ROOT}/03_long_text_rendering/multiseed/${s}/config.yaml"
  mkdir -p "$(dirname "$SEED_CONFIG")"
  python - "$CONFIG" "$SEED_CONFIG" "$s" "$ANCHOR_SC" "$ANCHOR_V" <<'PY'
import sys, yaml
base = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
out, seed, anchor_sc, anchor_v = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
from pathlib import Path
scope = str(Path(out).parent)
base["train"]["seed"] = int(seed)
base["checkpoints"]["paper_selfcond_step"] = int(anchor_sc)
base["checkpoints"]["paper_vanilla_step"] = int(anchor_v)
base["checkpoints"]["vanilla_dir"] = f"{scope}/vanilla"
base["checkpoints"]["selfcond_dir"] = f"{scope}/selfcond"
base["outputs"]["pair_manifest"] = f"{scope}/pairs.json"
base["outputs"]["audit_manifest"] = f"{scope}/audit.json"
base["outputs"]["eval_root"] = f"{scope}/eval"
base["outputs"]["summary"] = f"{scope}/summary.json"
base["outputs"]["figure"] = f"{scope}/cer_curves.svg"
base["outputs"]["cer_curves"] = f"{scope}/cer_curves.json"
with open(out, "w", encoding="utf-8") as f:
    yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)
print(out)
PY
  echo "[run_multiseed] seed $s (anchor SelfCond@$ANCHOR_SC) -> $SEED_CONFIG"
  CONFIG_OVERRIDE="$SEED_CONFIG" PROFILE="$PROFILE" bash "$SELF_DIR/run.sh" \
    --profile "$PROFILE" \
    --stage 03_train_vanilla --stage 04_train_selfcond --stage 05_match_checkpoints \
    --stage 06_evaluate --stage 07_summarize
  SEED_CONFIGS+=("$SEED_CONFIG")
done

python - "$OUTPUT_ROOT/03_long_text_rendering/multiseed" "${SEED_CONFIGS[@]}" <<'PY'
import json, sys
from pathlib import Path
from statistics import mean, stdev

root = Path(sys.argv[1])
configs = [Path(c) for c in sys.argv[2:]]
rows = []
for cfg in configs:
    sp = root / cfg.parent.name / "summary.json"
    rows.append({"seed": cfg.parent.name,
                 "summary": json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else None})

agg = {"experiment": "long_text_rendering", "seeds": sorted(set(r["seed"] for r in rows)), "anchor": None,
       "note": "PAPER REPORTS NO MULTI-SEED NUMBERS for this task; this is a local extension only.",
       "methods": {}}
for method in ("vanilla", "selfcond"):
    cer_means, perfects = [], []
    for r in rows:
        s = r["summary"]
        if s:
            agg["anchor"] = (s.get("pair") or {}).get("paper_selfcond_step") or agg["anchor"]
            cell = s.get("eval", {}).get(method) or {}
            if cell.get("cer_mean") is not None:
                cer_means.append(cell["cer_mean"])
            if cell.get("perfect_frac") is not None:
                perfects.append(cell["perfect_frac"])
    agg["methods"][method] = {
        "n_seeds_scored": len(cer_means),
        "cer_mean_mean": mean(cer_means) if cer_means else None,
        "cer_mean_sd": stdev(cer_means) if len(cer_means) > 1 else 0.0,
        "perfect_frac_mean": mean(perfects) if perfects else None,
    }

agg["complete"] = all(cell["n_seeds_scored"] == len(configs) for cell in agg["methods"].values())
out = root / "multiseed_summary.json"
out.write_text(json.dumps(agg, indent=2), encoding="utf-8")
print(json.dumps(agg, indent=2))
if not agg["complete"]:
    raise SystemExit("Missing seed scores: multi-seed aggregate is incomplete")
PY

echo "[run_multiseed] done -> $OUTPUT_ROOT/03_long_text_rendering/multiseed/multiseed_summary.json"
