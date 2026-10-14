#!/usr/bin/env python
"""FLOP-aligned eval curve for the long-multiplication runs, rejudged data.

Runs compared (all in <WANDB_ENTITY>/imagereasoning-exp-train, both trained
on data/long_multiplication/data_200k and periodically evaluated on the same
held-out data_test100 split, NFE 50, strict judge, 100 samples):

Original training runs (source of the cumulative-FLOPs x-axis):
    vanilla  <wandb-run-id>  longmul_vanilla_15ep   (15 epochs, eval every 12500 steps)
    selfcond <wandb-run-id>  longmul_selfcond_5ep   (5 epochs,  eval every 5000 steps)

Rejudge runs (source of the metric): the same periodic-eval predictions, rejudged
with the tightened judge prompt (src/image_only/eval_prompt_longmul.txt):
    vanilla  <wandb-run-id>  longmul_vanilla_15ep_rejudge_newprompt
    selfcond <wandb-run-id>  longmul_selfcond_5ep_rejudge_newprompt
They carry only eval/* keys (no train/cumulative_flops), so their steps are
joined with the training runs' FLOPs by global step.

Both methods fine-tune FLUX.2-klein-base-4B with LoRA (rank 32) and differ only
in method, so FLOPs differ purely by the per-micro-batch rate.

X-axis: cumulative FLOPs as logged by the training runs themselves from their
measured poke rates (cumulative_flops = rate * global_step * grad_accum *
num_processes). Both methods read `train/cumulative_flops` from their
periodic-eval rows.

Metric: accuracy = eval/judge_num_correct / 100, i.e. the fraction of the full
100-sample eval set that the strict judge accepts. Unlike eval/judge_accuracy
(which divides only by the valid responses and thus drops judge errors), this
counts every judge error / unparseable response as incorrect.

FLOP budget: both curves are truncated at the vanilla checkpoint nearest
2.3e19 cumulative FLOPs (step 312,500 at 2.324e19), the comparison budget used
in the paper figure; vanilla ran on past it (15 epochs, 375,000 steps), but
those later checkpoints are not plotted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Repo root (this script lives at <repo>/src/analysis/long_multiplication/curve_analysis.py).
ROOT = Path(__file__).resolve().parents[3]
TASK_SLUG = "long_multiplication"

# W&B entity (account) that owns these runs; the account name is not part of this
# repository. Set WANDB_ENTITY to your own W&B account (cf. pack_cherry.py).
PROJECT = f"{os.environ['WANDB_ENTITY']}/imagereasoning-exp-train"
# Original training runs: they stamp cumulative FLOPs onto their periodic-eval
# rows under train/.
TRAIN_RUNS = {
    "vanilla": "<wandb-run-id>",     # longmul_vanilla_15ep
    "selfcond": "<wandb-run-id>",    # longmul_selfcond_5ep
}
# Rejudge runs: the same periodic-eval predictions rejudged with the tightened
# judge prompt; they carry the final judge_num_correct numbers but no FLOPs.
JUDGE_RUNS = {
    "vanilla": "<wandb-run-id>",     # longmul_vanilla_15ep_rejudge_newprompt
    "selfcond": "<wandb-run-id>",    # longmul_selfcond_5ep_rejudge_newprompt
}
FLOPS_KEY = "train/cumulative_flops"
NUM_CORRECT_KEY = "eval/judge_num_correct"
NUM_EVAL_KEY = "eval/num_samples"        # 100
ACC_KEY = "eval/judge_accuracy"          # kept only for reference; NOT the metric
STEP_KEY = "_step"


def pull_train_flops(api, method: str) -> dict[int, float]:
    """Map global step -> cumulative FLOPs from the original training run."""
    run = api.run(f"{PROJECT}/{TRAIN_RUNS[method]}")
    rows = list(run.scan_history(keys=[STEP_KEY, FLOPS_KEY]))
    flops_by_step: dict[int, float] = {}
    for r in rows:
        fl = r.get(FLOPS_KEY)
        if fl is None:
            continue
        flops_by_step[int(r[STEP_KEY])] = float(fl)
    return flops_by_step


def pull_judge_results(api, method: str) -> list[dict]:
    """Pull one row per evaluated step from the rejudge run."""
    run = api.run(f"{PROJECT}/{JUDGE_RUNS[method]}")
    rows = list(run.scan_history(
        keys=[STEP_KEY, NUM_CORRECT_KEY, NUM_EVAL_KEY, ACC_KEY]))
    # Eval rows can be duplicated in the history; keep one row per step.
    by_step: dict[int, dict] = {}
    for r in rows:
        nc = r.get(NUM_CORRECT_KEY)
        n = r.get(NUM_EVAL_KEY)
        if nc is None or n is None or n == 0:
            continue
        by_step[int(r[STEP_KEY])] = {
            "step": int(r[STEP_KEY]),
            "num_correct": int(nc),
            "num_samples": int(n),
            "judge_accuracy": r.get(ACC_KEY),  # reference only; NOT the metric
        }
    return [by_step[s] for s in sorted(by_step)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("WANDB_API_KEY"),
                    help="wandb API key (default: $WANDB_API_KEY)")
    ap.add_argument("--out-csv", default=str(ROOT / "outputs" / TASK_SLUG / "curves.csv"))
    ap.add_argument("--out-png", default=str(ROOT / "outputs" / TASK_SLUG / "eval_vs_flops.png"))
    ap.add_argument("--out-json", default=str(ROOT / "outputs" / TASK_SLUG / "curves_aligned.json"))
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    import wandb
    api = wandb.Api(api_key=args.api_key or os.environ.get("WANDB_API_KEY"))

    data = {}
    for m in TRAIN_RUNS:
        flops_by_step = pull_train_flops(api, m)
        judge_rows = pull_judge_results(api, m)
        joined = []
        for h in judge_rows:
            fl = flops_by_step.get(h["step"])
            if fl is None:
                print(f"[warn] {m}: step {h['step']} rejudged but has no "
                      f"cumulative FLOPs in the training run; skipped")
                continue
            # accuracy = judge_num_correct / 100 (full eval set), NOT judge_accuracy.
            joined.append({
                "step": h["step"],
                "flops": fl,
                "accuracy": float(h["num_correct"]) / 100.0,
                "num_correct": h["num_correct"],
                "num_samples": h["num_samples"],
            })
        joined.sort(key=lambda h: h["step"])
        data[m] = joined
        print(f"[pull] {m}: {len(joined)} eval points, "
              f"flops[{joined[0]['step']}..{joined[-1]['step']}] = "
              f"{joined[0]['flops']:.3e}..{joined[-1]['flops']:.3e}, "
              f"acc[{joined[0]['step']}..{joined[-1]['step']}] = "
              f"{joined[0]['accuracy']:.3f}..{joined[-1]['accuracy']:.3f}")

    # --- FLOPs alignment / truncation ---
    # Comparison budget = the vanilla checkpoint nearest 2.3e19 cumulative FLOPs
    # (step 312,500). Vanilla trained past it; those checkpoints are not plotted.
    vanilla_full = data["vanilla"]
    budget_flops = min(vanilla_full, key=lambda h: abs(h["flops"] - 2.3e19))["flops"]
    vanilla_trunc = [h for h in vanilla_full if h["flops"] <= budget_flops]
    sc_full = data["selfcond"]
    sc_trunc = [h for h in sc_full if h["flops"] <= budget_flops]
    print(f"[align] budget flops = {budget_flops:.3e} (vanilla step "
          f"{min(vanilla_full, key=lambda h: abs(h['flops'] - budget_flops))['step']})")
    print(f"[align] vanilla truncated {len(vanilla_full) - len(vanilla_trunc)}/"
          f"{len(vanilla_full)} points; kept {len(vanilla_trunc)}")
    print(f"[align] selfcond truncated {len(sc_full) - len(sc_trunc)}/"
          f"{len(sc_full)} points (flops > budget); kept {len(sc_trunc)}")

    vanilla_trunc.sort(key=lambda h: h["flops"])
    sc_trunc.sort(key=lambda h: h["flops"])
    data["vanilla"] = vanilla_trunc
    data["selfcond"] = sc_trunc

    # --- persist ---
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "step", "pass_rate", "cumulative_flops"])
        for m in TRAIN_RUNS:
            for h in data[m]:
                w.writerow([m, h["step"], f"{h['accuracy']:.6f}",
                            f"{h['flops']:.6e}"])
    payload = {
        "config": {
            "metric": "eval/judge_num_correct / 100 (strict whole-worksheet judge, over the full 100-sample set)",
            "eval_split": "data/long_multiplication/data_test100 (both methods)",
            "judge_runs": JUDGE_RUNS,
            "train_runs": TRAIN_RUNS,
            "vanilla_max_flops": budget_flops,
        },
        "runs": {m: data[m] for m in TRAIN_RUNS},
        "vanilla_alignment": {
            "total": len(vanilla_full), "kept": len(vanilla_trunc),
            "dropped": len(vanilla_full) - len(vanilla_trunc),
        },
        "selfcond_alignment": {
            "total": len(sc_full), "kept": len(sc_trunc),
            "dropped": len(sc_full) - len(sc_trunc),
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
                                    [h["accuracy"] for h in data["vanilla"]])
    xs["selfcond"], ys["selfcond"] = ([h["flops"] for h in sc_trunc],
                                      [h["accuracy"] for h in sc_trunc])

    # Clean budget tick: round the budget to 2 decimals in x10^18 units.
    budget_label = f"{budget_flops / 1e18:.2f}"   # e.g. "23.24"
    budget_clean = float(budget_label) * 1e18
    # Round the axis top to the next multiple of 5 (x10^18) so it does not leave
    # a large empty tail beyond the last data point.
    xlim_top = math.ceil(budget_clean / 1e18 / 5) * 5 * 1e18

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs["vanilla"], ys["vanilla"], marker="o", ms=4, lw=1.8,
            label="VanillaFM", color="#1f77b4")
    ax.plot(xs["selfcond"], ys["selfcond"], marker="s", ms=4, lw=1.8,
            label="SelfCondFM", color="#d62728")
    ax.axvline(budget_clean, color="gray", ls="--", lw=1,
               label="comparison FLOPs budget")
    ax.set_xlabel("Cumulative FLOPs ($\\times 10^{18}$)")
    tick_step = 5 if xlim_top / 1e18 <= 30 else 10
    xtick_vals = [float(i) * 1e18 for i in range(0, int(xlim_top / 1e18), tick_step)]
    xtick_labs = [str(i) for i in range(0, int(xlim_top / 1e18), tick_step)]
    xtick_vals.append(budget_clean)
    xtick_labs.append(budget_label)
    ax.set_xticks(xtick_vals)
    ax.set_xticklabels(xtick_labs)
    ax.set_xlim(0, xlim_top)
    ax.set_ylabel("Whole-Worksheet Strict Judge Accuracy")
    ax.set_title("Long multiplication: Whole-Worksheet Strict Judge Accuracy vs FLOPs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"[out] png -> {args.out_png}")


if __name__ == "__main__":
    main()
