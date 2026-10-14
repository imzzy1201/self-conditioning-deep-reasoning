#!/usr/bin/env python
"""DEPRECATED — historical source. Use `experiment_tools/flops_cli.py match`.

This legacy selector predates the common `match_pairs` implementation. Kept only
as a historical provenance reference; the active pipeline
routes through `flops_cli.py match` (`src/experiment_tools/flops.py`).

The original runs saved a checkpoint every 1,000 optimizer steps.  A single-GPU
poke measures one replica's fwd+bwd FLOPs at the original per-GPU batch size;
the original four-GPU run's cumulative FLOPs are then

  rate_per_micro_batch * step * grad_accum * 4.

For each SelfCond checkpoint it matches the nearest Vanilla checkpoint whose
cumulative FLOPs are at least as large (>=) and selects the highest SelfCond
compute budget. No relative-gap threshold is applied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla_rate", default="outputs/line_tracing/poke/vanilla/flops.json")
    parser.add_argument("--selfcond_rate", default="outputs/line_tracing/poke/selfcond/flops.json")
    parser.add_argument("--vanilla_max_step", type=int, default=60_000)
    parser.add_argument("--selfcond_max_step", type=int, default=53_000)
    parser.add_argument("--checkpoint_interval", type=int, default=1_000)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--original_num_processes", type=int, default=4)
    parser.add_argument(
        "--output",
        default="outputs/line_tracing/line_tracing_9l_selected_pair.json",
    )
    return parser.parse_args()


def load_rate(path: str) -> float:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["per_micro_batch_flops"])


def main() -> None:
    args = parse_args()
    if args.checkpoint_interval <= 0 or args.original_num_processes <= 0:
        raise SystemExit("checkpoint interval and process count must be positive")

    vanilla_rate = load_rate(args.vanilla_rate)
    selfcond_rate = load_rate(args.selfcond_rate)
    multiplier = args.grad_accum * args.original_num_processes
    vanilla = [
        {"step": step, "cumulative_flops": vanilla_rate * step * multiplier}
        for step in range(args.checkpoint_interval, args.vanilla_max_step + 1, args.checkpoint_interval)
    ]
    selfcond = [
        {"step": step, "cumulative_flops": selfcond_rate * step * multiplier}
        for step in range(args.checkpoint_interval, args.selfcond_max_step + 1, args.checkpoint_interval)
    ]

    candidates = []
    for sc in selfcond:
        match = next((v for v in vanilla if v["cumulative_flops"] >= sc["cumulative_flops"]), None)
        if match is None:
            continue
        gap = match["cumulative_flops"] - sc["cumulative_flops"]
        candidates.append(
            {
                "selfcond_step": sc["step"],
                "vanilla_step": match["step"],
                "selfcond_flops": sc["cumulative_flops"],
                "vanilla_flops": match["cumulative_flops"],
                "absolute_gap_flops": gap,
                "relative_gap": gap / sc["cumulative_flops"],
            }
        )

    if not candidates:
        raise SystemExit("No Vanilla checkpoint reaches any SelfCond checkpoint's FLOPs")
    chosen = max(candidates, key=lambda c: c["selfcond_flops"])
    output = {
        "selection_rule": (
            "For every SelfCond checkpoint, use the nearest Vanilla checkpoint with "
            "at least as much cumulative FLOPs (>=); select the highest SelfCond FLOP "
            "budget. No relative-gap threshold."
        ),
        "measurement": {
            "vanilla_per_micro_batch_flops_one_process": vanilla_rate,
            "selfcond_per_micro_batch_flops_one_process": selfcond_rate,
            "grad_accum": args.grad_accum,
            "original_num_processes": args.original_num_processes,
            "cluster_total_multiplier": multiplier,
        },
        "chosen_pair": chosen,
        "candidate_pairs": candidates,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(chosen, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
