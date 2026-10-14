#!/usr/bin/env python
"""FLOP-aligned eval curve for the a5-line-trace-image runs.

Pulls the vanilla (baseline) and selfcond (proj_condition) runs from wandb,
aligns them on cumulative FLOPs using the Part-1 estimation protocol
(scripts/01_line_tracing + src/experiment_tools/flops.py), and
plots eval/max_correct_prefix_steps.mean vs cumulative FLOPs. The selfcond
curve is truncated at the vanilla run's maximum cumulative FLOPs.

FLOP alignment formula (from flops.py / 02_measure_flops.sh):
    cumulative_flops = per_micro_batch_flops * global_step
                       * gradient_accumulation_steps * num_processes
per_micro_batch_flops is the MEASURED fwd+bwd FLOPs of one micro-batch on ONE
process (batch size baked in), taken from a 100-micro-batch --measure_flops poke
(flops.json). The measured rates used here are in
outputs/line_tracing/poke/{vanilla,selfcond}/flops.json (vanilla 1.875e13,
selfcond 2.106e13). No analytic estimate is used — pass --flops-json to point at
other poke output dirs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Repo root (this script lives at <repo>/src/analysis/line_tracing/curve_analysis.py).
ROOT = Path(__file__).resolve().parents[3]
TASK_SLUG = "line_tracing"

# W&B entity (account) that owns these runs; the account name is not part of this
# repository. Set WANDB_ENTITY to your own W&B account (cf. pack_cherry.py).
PROJECT = f"{os.environ['WANDB_ENTITY']}/a5-line-trace-image"
RUNS = {
    "vanilla": "<wandb-run-id>",     # baseline_9L_4gpu_50
    "selfcond": "<wandb-run-id>",    # proj_condition_9L_4gpu_50
}
EVAL_KEY = "eval/max_correct_prefix_steps.mean"
STEP_KEY = "_step"

# --- Run hyper-parameters (identical for both runs, from wandb config) -------
BATCH_SIZE = 8           # per-GPU / per-micro-batch
GRAD_ACCUM = 1           # gradient_accumulation_steps
NUM_PROCESSES = 4        # "4gpu"

# --- 9M_LONG from-scratch FLUX.2 transformer (src/image_only/configs.py) -------------
NUM_LAYERS = 3
NUM_SINGLE_LAYERS = 3
NUM_ATTN_HEADS = 4
ATTN_HEAD_DIM = 64
MLP_RATIO = 3.0
IN_CHANNELS = 128
OUT_CHANNELS = 128
SEQ_LEN = 1005           # 5 rows x (4*50+1) image tokens, n=50 stride=4 patch=16

# Documented no-text selfcond/vanilla per-micro-batch ratio (flops.py): selfcond
# does 2 fwd+bwd passes (2-then-3 token blocks), vanilla does 1 (2 blocks);
# linear-dominated limit -> 1 + 3/2 = 2.5.
RATIO_SELF_VANILLA = 2.5


def estimate_per_micro_batch_flops(batch_size: int = BATCH_SIZE) -> float:
    """Analytic fwd+bwd FLOPs of one micro-batch of the 9M_LONG transformer.

    FLUX.2 blocks: each of (num_layers + num_single_layers) blocks has one
    self-attention + one MLP over the image tokens (no-text -> empty context).
    Per token, per block, fwd+bwd (fwd+bwd ~= 3x forward):
        attention: 12*d**2            (QKV + out proj, 3x)
                 + 6*L*d              (QK^T and score@V, 3x)
        mlp:       6*d*d_mlp          (two linear layers, 3x)
    Plus patch embed (in->d) and output proj (d->out), small.
    Sequence: batch_size * SEQ_LEN tokens.
    """
    d = NUM_ATTN_HEADS * ATTN_HEAD_DIM          # inner_dim = 256
    d_mlp = int(d * MLP_RATIO)                  # 768
    n_blocks = NUM_LAYERS + NUM_SINGLE_LAYERS   # 6
    per_tok = n_blocks * (12 * d * d + 6 * SEQ_LEN * d + 6 * d * d_mlp)
    per_tok += 3 * IN_CHANNELS * d              # patch embed fwd+bwd
    per_tok += 3 * d * OUT_CHANNELS             # output proj fwd+bwd
    tokens = batch_size * SEQ_LEN
    return float(per_tok * tokens)


DEFAULT_FLOP_JSON = ROOT / "outputs/line_tracing/poke"


def load_flops_rate(method: str, flops_json_cli: str | None) -> float:
    """Return the per-micro-batch rate for a method from a MEASURED flops.json.

    The protocol requires a measured rate from a 100-micro-batch --measure_flops
    poke (see 02_measure_flops.sh). This function hard-fails if no measured
    flops.json is found for the method — no analytic fallback.
    """
    base = Path(flops_json_cli or DEFAULT_FLOP_JSON)
    p = base if base.is_file() else base / method / "flops.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"No measured flops.json for '{method}' at {p}. Run the 100-micro-batch "
            "--measure_flops poke first (src/image_only/train.py --measure_flops) and pass "
            "its output dir via --flops-json."
        )
    with open(p, encoding="utf-8") as f:
        fj = json.load(f)
    rate = float(fj["per_micro_batch_flops"])
    print(f"[flops] {method}: MEASURED rate {rate:.3e} from {p}")
    return rate


def cumulative_flops(rate: float, global_step: int) -> float:
    """Cluster-total cumulative FLOPs at an optimizer step (Part-1 formula)."""
    return rate * global_step * GRAD_ACCUM * NUM_PROCESSES


def pull_history(api, method: str) -> list[dict]:
    run = api.run(f"{PROJECT}/{RUNS[method]}")
    rows = list(run.scan_history(keys=[STEP_KEY, EVAL_KEY]))
    out = []
    for r in rows:
        val = r.get(EVAL_KEY)
        if val is None:
            continue
        out.append({
            "step": int(r[STEP_KEY]),
            "eval": float(val),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY"),
                    help="wandb API key (default: $WANDB_API_KEY)")
    ap.add_argument("--flops-json", default=DEFAULT_FLOP_JSON,
                    help="dir of measured flops.json poke outputs (default: outputs/line_tracing/poke)")
    ap.add_argument("--out-csv", default=str(ROOT / "outputs" / TASK_SLUG / "curves.csv"))
    ap.add_argument("--out-png", default=str(ROOT / "outputs" / TASK_SLUG / "eval_vs_flops.png"))
    ap.add_argument("--out-json", default=str(ROOT / "outputs" / TASK_SLUG / "curves_aligned.json"))
    args = ap.parse_args()

    import wandb
    api = wandb.Api(api_key=args.api_key)

    rates = {m: load_flops_rate(m, args.flops_json) for m in RUNS}
    data = {}
    for m in RUNS:
        hist = pull_history(api, m)
        for h in hist:
            h["flops"] = cumulative_flops(rates[m], h["step"])
        data[m] = hist
        n = len(hist)
        print(f"[pull] {m}: {n} eval points, "
              f"flops[{hist[0]['step']}..{hist[-1]['step']}] = "
              f"{hist[0]['flops']:.3e}..{hist[-1]['flops']:.3e}")

    # --- FLOPs alignment / truncation: drop selfcond points beyond vanilla max ---
    vanilla_max_flops = max(h["flops"] for h in data["vanilla"])
    sc_full = data["selfcond"]
    sc_trunc = [h for h in sc_full if h["flops"] <= vanilla_max_flops]
    print(f"[align] vanilla max flops = {vanilla_max_flops:.3e}")
    print(f"[align] selfcond truncated {len(sc_full) - len(sc_trunc)}/{len(sc_full)} "
          f"points (flops > vanilla max); kept {len(sc_trunc)}")

    # sort ascending by flops for clean plotting
    for m in RUNS:
        data[m].sort(key=lambda h: h["flops"])
    sc_trunc.sort(key=lambda h: h["flops"])
    # persist the ALIGNED (truncated) selfcond curve, matching the plot
    data["selfcond"] = sc_trunc

    # --- persist ---
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "step", "eval/max_correct_prefix_steps.mean", "cumulative_flops"])
        for m in RUNS:
            for h in data[m]:
                w.writerow([m, h["step"], h["eval"], f"{h['flops']:.6e}"])
    # selfcond is stored truncated (aligned); counts of what was dropped
    payload = {
        "config": {
            "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
            "num_processes": NUM_PROCESSES,
            "per_micro_batch_flops": rates,
            "vanilla_max_flops": vanilla_max_flops,
        },
        "runs": {m: data[m] for m in RUNS},
        "selfcond_alignment": {"total": len(sc_full), "kept": len(sc_trunc), "dropped": len(sc_full) - len(sc_trunc)},
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[out] csv -> {args.out_csv}")
    print(f"[out] json -> {args.out_json}")

    # --- plot ---
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    xs, ys = {}, {}
    xs["vanilla"], ys["vanilla"] = [h["flops"] for h in data["vanilla"]], [h["eval"] for h in data["vanilla"]]
    xs["selfcond"], ys["selfcond"] = [h["flops"] for h in sc_trunc], [h["eval"] for h in sc_trunc]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs["vanilla"], ys["vanilla"], marker="o", ms=4, lw=1.8,
            label="VanillaFM", color="#1f77b4")
    ax.plot(xs["selfcond"], ys["selfcond"], marker="s", ms=4, lw=1.8,
            label="SelfCondFM", color="#d62728")
    # Mark the vanilla max with a clean "4.50" tick (exact max 4.500545e18
    # rounds to 4.50e18). Put BOTH the dashed line and the "4.50" tick at the
    # same clean 4.50e18 position so the label and line coincide exactly.
    vanilla_max_clean = 4.50e18
    ax.axvline(vanilla_max_clean, color="gray", ls="--", lw=1,
               label="comparison FLOPs budget")
    ax.set_xlabel("Cumulative FLOPs ($\\times 10^{18}$)")
    ax.set_xticks([0, 1e18, 2e18, 3e18, 4e18, vanilla_max_clean])
    ax.set_xticklabels(["0", "1", "2", "3", "4", "4.50"])
    ax.set_ylabel("Correct Trace Depth (Average)")
    ax.set_title("Line tracing, n=50: Correct Trace Depth vs FLOPs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"[out] png -> {args.out_png}")


if __name__ == "__main__":
    main()