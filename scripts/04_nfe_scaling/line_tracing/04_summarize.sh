#!/usr/bin/env bash
# 04_summarize.sh — aggregate the two methods' cells into ONE table for both axes.
#
# Reads each method's eval summary (produced by 03) and writes the single
# cells.csv that downstream fig:nfe_scaling (x = NFE) and fig:inference_flops
# (x = transformer-forward FLOPs per sample) both consume — it is never
# re-evaluated per axis. Also writes inference_flops.json (per method/NFE/seed
# cell FLOPs) and summary.json (n_expected/n_scored/missing + both-axis curves).
# Missing cells are marked `incomplete`, never zero-filled or silently dropped.
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

EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"
CELLS_CSV="$(resolve_path "$(CFG outputs.cells_csv)")"
INFERENCE_FLOPS="$(resolve_path "$(CFG outputs.inference_flops)")"
SUMMARY_OUT="$(resolve_path "$(CFG outputs.summary)")"
NFE_MIN="$(CFG eval.nfe_min 1)"
NFE_MAX="$(CFG eval.nfe_max 50)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$CONFIG" "$PAIR_MANIFEST" "$EVAL_METADATA" "$EVAL_ROOT" "$CELLS_CSV" \
  "$INFERENCE_FLOPS" "$SUMMARY_OUT" "$NFE_MIN" "$NFE_MAX" <<'PY'
import csv, json, sys
from pathlib import Path

cfg_p, pair_p = Path(sys.argv[1]), Path(sys.argv[2])
id_manifest = str(Path(sys.argv[3]))
eval_root = Path(sys.argv[4])
cells_csv, flops_p, summary_p = Path(sys.argv[5]), Path(sys.argv[6]), Path(sys.argv[7])
nfe_min, nfe_max = int(sys.argv[8]), int(sys.argv[9])

methods = ("selfcond", "vanilla")
pair = json.loads(pair_p.read_text(encoding="utf-8")) if pair_p.is_file() else {}
checkpoints = pair.get("checkpoints", {})

FIELDNAMES = ["method", "checkpoint", "NFE", "noise_seed", "IDmanifest",
              "n_total", "LCP", "FLOPs_per_sample", "call_count", "status", "source"]

rows, flops_cells, missing, warnings = [], [], [], []
per_method = {m: {} for m in methods}

for method in methods:
    summary_f = eval_root / method / "summary.json"
    if not summary_f.is_file():
        missing.append(f"{method}: summary.json absent (run 03_sweep_nfe)")
        continue
    sweep = json.loads(summary_f.read_text(encoding="utf-8"))
    checkpoint = checkpoints.get(method, {}).get("checkpoint", "")
    for cell in sweep.get("cells", []):
        nfe, seed = int(cell["nfe"]), int(cell["seed"])
        metrics = cell.get("metrics") or {}
        flops = cell.get("flops") or {}
        cell_dir = f"{method}/nfe_{nfe:03d}/seed_{seed:04d}"
        status = "ok" if metrics and flops else "incomplete"
        if not metrics:
            missing.append(f"{cell_dir}: no metrics.json")
        if not flops:
            missing.append(f"{cell_dir}: no inference_flops.json")
        n_total = metrics.get("num_samples")
        if n_total not in (None, "") and int(n_total) != 200:
            warnings.append(f"{cell_dir}: n_total={n_total}, expected 200")
        lcp = metrics.get("max_correct_prefix_steps.mean")
        flops_per_sample = flops.get("transformer_forward_flops_per_sample")
        rows.append({
            "method": method,
            "checkpoint": checkpoint,
            "NFE": nfe,
            "noise_seed": seed,
            "IDmanifest": id_manifest,
            "n_total": "" if n_total is None else n_total,
            "LCP": "" if lcp is None else lcp,
            "FLOPs_per_sample": "" if flops_per_sample is None else flops_per_sample,
            "call_count": flops.get("num_inference_steps", ""),
            "status": status,
            "source": cell_dir,
        })
        per_method[method][str(nfe)] = {
            "LCP": lcp,
            "FLOPs_per_sample": flops_per_sample,
            "call_count": flops.get("num_inference_steps"),
        }
        if flops:
            flops_cells.append({
                "method": method, "NFE": nfe, "seed": seed,
                "num_inference_steps": flops.get("num_inference_steps"),
                "num_samples": flops.get("num_samples"),
                "transformer_forward_flops_total": flops.get("transformer_forward_flops_total"),
                "transformer_forward_flops_per_sample": flops_per_sample,
                "source": f"{cell_dir}/inference_flops.json",
            })

# cells.csv — the single table both axes read.
cells_csv.parent.mkdir(parents=True, exist_ok=True)
with cells_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    w.writerows(rows)

# inference_flops.json — per-cell transformer-forward FLOPs records.
flops_report = {
    "status": "ok" if flops_cells else "incomplete",
    "n_cells": len(flops_cells),
    "n_expected_cells": 2 * (nfe_max - nfe_min + 1) * 1,
    "scope": "transformer backbone forward FLOPs per sample (torch FlopCounterMode); "
             "VAE/scheduler excluded; entirely separate from the training rate "
             "(docs/flops_protocol.md).",
    "cells": flops_cells,
    "missing": missing,
}
flops_p.parent.mkdir(parents=True, exist_ok=True)
flops_p.write_text(json.dumps(flops_report, indent=2), encoding="utf-8")

# summary.json — provenance + both-axis curves from the SAME cells.
n_scored = sum(1 for r in rows if r["status"] == "ok")
summary = {
    "experiment": "nfe_scaling",
    "paper_labels": ["fig:nfe_scaling", "fig:inference_flops"],
    "config": str(cfg_p),
    "pair": str(pair_p),
    "pair_consumed_as_is": pair.get("consumed_as_is", True),
    "eval_metadata": id_manifest,
    "nfe_grid": [nfe_min, nfe_max],
    "n_expected_cells": 2 * (nfe_max - nfe_min + 1) * 1,
    "n_scored_cells": n_scored,
    "metric": "max_correct_prefix_steps.mean (LCP) over 200 fixed instances",
    "axes": "fig:nfe_scaling (x=NFE) and fig:inference_flops (x=FLOPs_per_sample) "
            "share cells.csv; LCP is the y-axis for both.",
    "per_method": per_method,
    "warnings": warnings,
    "missing": missing,
    "figures": "rendered downstream from cells.csv (src/analysis/, not yet migrated); "
               "this stage emits the data files only.",
}
summary_p.parent.mkdir(parents=True, exist_ok=True)
summary_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
if missing:
    print(f"[04_summarize] {len(missing)} incomplete cell(s), see summary.missing", file=sys.stderr)
if warnings:
    print(f"[04_summarize] {len(warnings)} warning(s), see summary.warnings", file=sys.stderr)
print(f"[04_summarize] done -> {cells_csv} (+ inference_flops.json, summary.json)")
PY

echo "[04_summarize] done"