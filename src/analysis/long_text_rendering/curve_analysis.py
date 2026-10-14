#!/usr/bin/env python
"""FLOP-aligned eval curve for the long-text-rendering runs (image_text, native text).

Runs compared (project `<WANDB_ENTITY>/imagereasoning-exp-train`):
    vanilla  <wandb-run-id>  longtext_vanilla_12ep
    selfcond <wandb-run-id>  longtext_selfcond_4ep
Both fine-tune FLUX.2-klein-base-4B with LoRA (rank 32) on TinyStories 300k
(384x384). They differ only in method, so FLOPs differ purely by the per-micro-
batch rate.

X-axis: `train/cumulative_flops` as logged by the runs themselves. Each run
recorded its *measured* per-micro-batch FLOPs rate (from a 100-micro-batch
--measure_flops poke,
scripts/03_long_text_rendering/02_measure_flops.sh)
into `flops_rate_source` and logged
    cumulative_flops = rate * global_step * grad_accum * num_processes
so the x-axis already carries the poke-based measurement — no local flops.json
is needed. (num_processes = 4: effective batch 16 = batch 1 x grad_accum 4 x 4.)

Metric: correct ratio = eval/cer_perfect_count / eval/num_samples (CER=0 perfect
renders over the 100-sample eval set), i.e. num_correct/100. The selfcond curve
is truncated at the vanilla run's maximum cumulative FLOPs (here a no-op, since
selfcond's 4-epoch run uses ~83% of vanilla's 12-epoch FLOP budget). The vanilla
tail is trimmed to the first eval point past selfcond's last (step 190,000 /
4.699e19), dropping the trailing points, so both curves end at a comparable FLOP
budget.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Repo root (this script lives at <repo>/src/analysis/long_text_rendering/curve_analysis.py).
ROOT = Path(__file__).resolve().parents[3]
TASK_SLUG = "long_text_rendering"

# W&B entity (account) that owns these runs; the account name is not part of this
# repository. Set WANDB_ENTITY to your own W&B account (cf. pack_cherry.py).
PROJECT = f"{os.environ['WANDB_ENTITY']}/imagereasoning-exp-train"
RUNS = {
    "vanilla": "<wandb-run-id>",     # longtext_vanilla_12ep
    "selfcond": "<wandb-run-id>",    # longtext_selfcond_4ep
}
CORRECT_KEY = "eval/cer_perfect_count"   # num_correct (CER=0 perfect renders)
NUM_EVAL_KEY = "eval/num_samples"        # 100
CF_KEY = "train/cumulative_flops"        # logged poke-based cumulative FLOPs
STEP_KEY = "_step"


def pull_history(api, method: str) -> list[dict]:
    run = api.run(f"{PROJECT}/{RUNS[method]}")
    rows = list(run.scan_history(
        keys=[STEP_KEY, CF_KEY, CORRECT_KEY, NUM_EVAL_KEY]))
    out = []
    for r in rows:
        cf = r.get(CF_KEY)
        perfect = r.get(CORRECT_KEY)
        n = r.get(NUM_EVAL_KEY)
        if cf is None or perfect is None or n is None or n == 0:
            continue
        out.append({
            "step": int(r[STEP_KEY]),
            "flops": float(cf),
            "correct": int(perfect),
            "num_correct_100": float(perfect) / float(n),  # correct ratio
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY"),
                    help="wandb API key (default: $WANDB_API_KEY)")
    ap.add_argument("--out-csv", default=str(ROOT / "outputs" / TASK_SLUG / "curves.csv"))
    ap.add_argument("--out-png", default=str(ROOT / "outputs" / TASK_SLUG / "eval_vs_flops.png"))
    ap.add_argument("--out-json", default=str(ROOT / "outputs" / TASK_SLUG / "curves_aligned.json"))
    args = ap.parse_args()

    import wandb
    api = wandb.Api(api_key=args.api_key)

    data = {}
    for m in RUNS:
        data[m] = pull_history(api, m)
        h = data[m]
        print(f"[pull] {m}: {len(h)} eval points, "
              f"flops[{h[0]['step']}..{h[-1]['step']}] = "
              f"{h[0]['flops']:.3e}..{h[-1]['flops']:.3e}")

    # --- FLOPs alignment / truncation ---
    # Vanilla tail trim: keep vanilla only through the FIRST eval point whose FLOPs
    # exceed SelfCondFM's last eval FLOPs, so both curves end at a comparable FLOP
    # budget (the trailing vanilla tail has no selfcond counterpart). Confirmed
    # cutoff: step 190,000 / 4.699e19 (drop steps 195k-225k).
    sc_last_flops = max(h["flops"] for h in data["selfcond"])
    vanilla_full = data["vanilla"]
    cutoff = next(h["flops"] for h in sorted(vanilla_full, key=lambda h: h["flops"])
                  if h["flops"] > sc_last_flops)
    vanilla_trunc = [h for h in vanilla_full if h["flops"] <= cutoff]
    print(f"[align] selfcond last eval flops = {sc_last_flops:.3e}")
    print(f"[align] vanilla tail cutoff = {cutoff:.3e} (next vanilla eval point "
          f"past selfcond); dropped {len(vanilla_full) - len(vanilla_trunc)}/"
          f"{len(vanilla_full)} tail points; kept {len(vanilla_trunc)}")
    vanilla_max_flops = cutoff   # budget line follows the trimmed vanilla max

    sc_full = data["selfcond"]
    sc_trunc = [h for h in sc_full if h["flops"] <= vanilla_max_flops]
    print(f"[align] vanilla max flops = {vanilla_max_flops:.3e}")
    print(f"[align] selfcond truncated {len(sc_full) - len(sc_trunc)}/"
          f"{len(sc_full)} points (flops > vanilla max); kept {len(sc_trunc)}")

    for m in RUNS:
        data[m].sort(key=lambda h: h["flops"])
    sc_trunc.sort(key=lambda h: h["flops"])
    data["vanilla"] = sorted(vanilla_trunc, key=lambda h: h["flops"])
    data["selfcond"] = sc_trunc

    # --- persist ---
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "step", "correct_ratio", "cumulative_flops"])
        for m in RUNS:
            for h in data[m]:
                w.writerow([m, h["step"], f"{h['num_correct_100']:.6f}",
                            f"{h['flops']:.6e}"])
    payload = {
        "config": {
            "metric": "eval/cer_perfect_count / eval/num_samples (correct ratio)",
            "vanilla_max_flops": vanilla_max_flops,
            "vanilla_tail_trim": "vanilla kept through first eval point past selfcond's last",
        },
        "runs": {m: data[m] for m in RUNS},
        "selfcond_alignment": {
            "total": len(sc_full), "kept": len(sc_trunc),
            "dropped": len(sc_full) - len(sc_trunc),
        },
        "vanilla_alignment": {
            "total": len(vanilla_full), "kept": len(vanilla_trunc),
            "dropped": len(vanilla_full) - len(vanilla_trunc),
        },
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[out] csv -> {args.out_csv}")
    print(f"[out] json -> {args.out_json}")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs, ys = {}, {}
    xs["vanilla"], ys["vanilla"] = ([h["flops"] for h in data["vanilla"]],
                                    [h["num_correct_100"] for h in data["vanilla"]])
    xs["selfcond"], ys["selfcond"] = ([h["flops"] for h in sc_trunc],
                                      [h["num_correct_100"] for h in sc_trunc])

    # Clean budget tick: round the vanilla max to 2 decimals in x10^18 units.
    budget_label = f"{vanilla_max_flops / 1e18:.2f}"   # e.g. "55.65"
    budget_clean = float(budget_label) * 1e18
    xlim_top = math.ceil(budget_clean / 1e18 / 10) * 10 * 1e18  # round up to x10

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs["vanilla"], ys["vanilla"], marker="o", ms=4, lw=1.8,
            label="VanillaFM", color="#1f77b4")
    ax.plot(xs["selfcond"], ys["selfcond"], marker="s", ms=4, lw=1.8,
            label="SelfCondFM", color="#d62728")
    ax.axvline(budget_clean, color="gray", ls="--", lw=1,
               label="comparison FLOPs budget")
    ax.set_xlabel("Cumulative FLOPs ($\\times 10^{18}$)")
    xtick_vals = [float(i) * 1e18 for i in range(0, int(xlim_top / 1e18), 10)]
    xtick_labs = [str(i) for i in range(0, int(xlim_top / 1e18), 10)]
    xtick_vals.append(budget_clean)
    xtick_labs.append(budget_label)
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labs)
    ax.set_xlim(0, xlim_top)
    ax.set_ylabel("Correct Ratio")
    ax.set_title("Long text rendering: Correct Ratio vs FLOPs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"[out] png -> {args.out_png}")


if __name__ == "__main__":
    main()