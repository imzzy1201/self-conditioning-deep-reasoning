#!/usr/bin/env bash
# 03_sweep_nfe.sh — run the NFE 1..50 sweep per method + record inference FLOPs.
#
# Consumes the frozen pair (02) and evaluates BOTH the SelfCond and Vanilla
# checkpoints over the 200-instance nfe_eval set. src/experiment_tools/
# eval_nfe_scaling.py runs inference (correct sampler via EXPERIMENT_METHOD) then
# scores CTD per (NFE, seed) cell; --measure_inference_flops writes the
# transformer-forward FLOPs (VAE/scheduler excluded) into each cell's
# inference_flops.json. A fixed base seed supplies identical per-instance noise
# for both methods at every NFE.
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

DATA_ROOT="$(resolve_path "$(CFG data.data_root)")"
EVAL_METADATA="$(resolve_path "$(CFG data.eval_metadata)")"
EVAL_ROOT="$(resolve_path "$(CFG outputs.eval_root)")"
PAIR_MANIFEST="$(resolve_path "$(CFG outputs.pair_manifest)")"
NFE_MIN="$(CFG eval.nfe_min 1)"
NFE_MAX="$(CFG eval.nfe_max 50)"
NOISE_SEED="$(CFG eval.noise_seed 0)"
MAX_SAMPLES="$(CFG eval.max_samples 200)"
MIFL="$(CFG eval.measure_inference_flops true)"
DTYPE="$(CFG eval.torch_dtype bf16)"
DEVICE="$(CFG eval.device cuda)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

python - "$REPO_ROOT" "$PAIR_MANIFEST" "$EVAL_METADATA" "$(dirname "$EVAL_METADATA")" "$EVAL_ROOT" \
  "$NFE_MIN" "$NFE_MAX" "$NOISE_SEED" "$MAX_SAMPLES" "$MIFL" "$DTYPE" "$DEVICE" <<'PY'
import json, subprocess, sys
from pathlib import Path

repo, pair_p = Path(sys.argv[1]), Path(sys.argv[2])
meta_p, data_p, eval_p = Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])
nfe_min, nfe_max = int(sys.argv[6]), int(sys.argv[7])
noise_seed, max_samples = int(sys.argv[8]), int(sys.argv[9])
measure_flops = sys.argv[10].lower() == "true"
dtype, device = sys.argv[11], sys.argv[12]

tool = repo / "src" / "experiment_tools" / "eval_nfe_scaling.py"
pair = json.loads(pair_p.read_text(encoding="utf-8"))
checkpoints = pair.get("checkpoints", {})
nfes = ",".join(str(n) for n in range(nfe_min, nfe_max + 1))

for method in ("selfcond", "vanilla"):
    if method not in checkpoints:
        raise SystemExit(f"frozen pair missing '{method}' (run 02_locate_checkpoints)")
    transformer = checkpoints[method]["transformer"]
    out = eval_p / method
    cmd = [sys.executable, str(tool),
           "--transformer_path", transformer,
           "--method", method,
           "--metadata_csv", str(meta_p),
           "--data_root", str(data_p),
           "--output_root", str(out),
           "--nfes", nfes,
           "--seeds", str(noise_seed),
           "--max_samples", str(max_samples),
           "--torch_dtype", dtype,
           "--device", device,
           "--measure_inference_flops" if measure_flops else "--no-measure_inference_flops"]
    print("[03_sweep_nfe]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo), check=True)

print(f"[03_sweep_nfe] done -> {eval_p}/{{selfcond,vanilla}}/summary.json")
PY

echo "[03_sweep_nfe] done"