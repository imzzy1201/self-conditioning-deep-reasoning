#!/usr/bin/env python
"""Plot available line-tracing NFE CTD results without fabricating missing points."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        default="outputs/line_tracing/nfe_dense_v60000_sc53000",
        help="Root containing vanilla/ and selfcond/ NFE result directories.",
    )
    parser.add_argument(
        "--output",
        default="outputs/line_tracing/nfe_dense_v60000_sc53000/nfe_ctd_partial.png",
    )
    return parser.parse_args()


def load_series(root: Path, method: str) -> tuple[list[int], list[float]]:
    points: list[tuple[int, float]] = []
    for metrics_path in (root / method).glob("nfe_*/seed_*/metrics.json"):
        match = re.fullmatch(r"nfe_(\d+)", metrics_path.parent.parent.name)
        if not match:
            continue
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        points.append((int(match.group(1)), float(metrics["max_correct_prefix_steps.mean"])))
    points.sort()
    return [x for x, _ in points], [y for _, y in points]


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    series = {
        "VanillaFM @ 60k": (*load_series(root, "vanilla"), "#666666", "o"),
        "SelfCondFM @ 53k": (*load_series(root, "selfcond"), "#377eb8", "s"),
    }
    if not any(xs for xs, _, _, _ in series.values()):
        raise SystemExit(f"No metrics.json files found below {root}")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for label, (xs, ys, color, marker) in series.items():
        if xs:
            ax.plot(xs, ys, marker=marker, linewidth=1.8, markersize=5,
                    color=color, label=label)
    ax.axhline(50, color="#999999", linestyle=":", linewidth=1)
    ax.text(50.2, 50, "maximum CTD", va="center", fontsize=8, color="#666666")
    ax.set(xlim=(1, 52), ylim=(0, 52), xlabel="Inference NFE", ylabel="Correct trace depth (CTD)")
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.grid(color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Line tracing NFE scaling — partial results", loc="left", fontweight="bold")
    fig.text(0.5, 0.01, "Matched pair: Vanilla@60000 vs SelfCond@53000; 50 n=50 test instances; seed 0.\nOnly completed points are shown.",
             ha="center", fontsize=8, color="#555555")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
