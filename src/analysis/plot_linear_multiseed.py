#!/usr/bin/env python3
"""Build three-seed diagnostic curves and paper figures.

Line tracing supports both the linear and original shifted inference schedules
and uses ``metrics.json``. Long multiplication uses the normal-schedule
``eval_periodic`` records produced during training, reconstructing pass rate
from ``judge_results.jsonl`` because those evaluations do not write
``metrics.json`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    from .plot_paper_figures import (
        BUDGET_COLOR,
        COLORS,
        LABELS,
        METHODS,
        NFE_BACKGROUND_COLOR,
        NFE_LEGEND_EDGE_COLOR,
        set_style,
        style_axis,
    )
except ImportError:  # executed directly from within src/analysis/
    from plot_paper_figures import (
        BUDGET_COLOR,
        COLORS,
        LABELS,
        METHODS,
        NFE_BACKGROUND_COLOR,
        NFE_LEGEND_EDGE_COLOR,
        set_style,
        style_axis,
    )


ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = ROOT / "outputs" / "figures"
SEEDS = (1201, 1202, 1203)
NUM_PROCESSES = 4
LINE_RATE = {"vanilla": 1.8752271712256e13, "selfcond": 2.1063824515072e13}
LONGMUL_RATE = {"vanilla": 9.29628776620032e12, "selfcond": 2.254634609793024e13}
LINE_NUM_SAMPLES = 200
LONGMUL_NUM_SAMPLES = 100
LONGMUL_VANILLA_MAX_STEPS = 125_000
LEGEND_EDGE = "#C8C8C8"
CURVE_WIDTH = 1.05
BUDGET_WIDTH = 0.9

LINE_EVAL_DIRS = {
    (1201, "vanilla"): ROOT / "outputs/a5_line_trace_image/baseline_9L_4gpu_50"
    / "eval_all_nfe50_seed1000_n200_lin",
    (1201, "selfcond"): ROOT / "outputs/a5_line_trace_image/proj_condition_9L_4gpu_50"
    / "eval_all_nfe50_seed1000_n200_lin",
    **{
        (seed, method): ROOT / "outputs" / "multi_seed" / "a5_line_trace"
        / f"{method}_seed{seed}" / "eval_all_nfe50_seed1000_n200_lin"
        for seed in (1202, 1203)
        for method in METHODS
    },
}
LINE_DEFAULT_EVAL_DIRS = {
    # These historical score directories are the default periodic evaluations:
    # 100 NFE, evaluation seed 1000, and the first 20 validation samples.
    # The reference directories correspond to the original seed 1201 runs.
    (1201, "vanilla"): ROOT / "outputs/multi_seed/a5_line_trace/scores"
    / "reference_vanilla_hist",
    (1201, "selfcond"): ROOT / "outputs/multi_seed/a5_line_trace/scores"
    / "reference_selfcond_hist",
    **{
        (seed, method): ROOT / "outputs/multi_seed/a5_line_trace/scores"
        / f"{method}_seed{seed}"
        for seed in (1202, 1203)
        for method in METHODS
    },
}
LINE_TRAIN_DIRS = {
    (1201, "vanilla"): ROOT / "outputs/a5_line_trace_image/baseline_9L_4gpu_50",
    (1201, "selfcond"): ROOT / "outputs/a5_line_trace_image/proj_condition_9L_4gpu_50",
    **{
        (seed, method): ROOT / "outputs" / "multi_seed" / "a5_line_trace"
        / f"{method}_seed{seed}"
        for seed in (1202, 1203)
        for method in METHODS
    },
}
LONGMUL_TRAIN_DIRS = {
    (1201, "vanilla"): ROOT / "outputs/long_multiplication/vanilla_new",
    (1201, "selfcond"): ROOT / "outputs/long_multiplication/selfcond_new",
    **{
        (seed, method): ROOT / "outputs" / "multi_seed_a100" / "long_multiplication"
        / f"{method}_seed{seed}"
        for seed in (1202, 1203)
        for method in METHODS
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_steps(root: Path) -> set[int]:
    steps = set()
    for path in root.glob("checkpoint-*"):
        tail = path.name.split("-", 1)[1]
        if tail.isdigit():
            steps.add(int(tail))
    return steps


def grad_accum(root: Path, expected: int) -> int:
    value = int(load_json(root / "train_args.json").get("gradient_accumulation_steps", 1))
    if value != expected:
        raise ValueError(f"{root}: expected grad_accum={expected}, found {value}")
    return value


def validate_steps(label: str, observed: set[int], train_dir: Path) -> None:
    expected = numeric_steps(train_dir)
    if observed != expected:
        raise ValueError(
            f"{label}: observed {len(observed)} eval steps but expected {len(expected)}; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def load_line_curves(
    eval_dirs: dict[tuple[int, str], Path] = LINE_EVAL_DIRS,
    *,
    num_samples: int = LINE_NUM_SAMPLES,
    strict_steps: bool = True,
) -> dict[tuple[int, str], list[dict]]:
    curves = {}
    for seed in SEEDS:
        for method in METHODS:
            eval_dir = eval_dirs[(seed, method)]
            accum = grad_accum(LINE_TRAIN_DIRS[(seed, method)], expected=1)
            rows = []
            # Both evaluator outputs (checkpoint-*) and scored periodic outputs
            # (step-*) are accepted here.
            for metrics_path in eval_dir.glob("*-*/metrics.json"):
                tail = metrics_path.parent.name.split("-", 1)[1]
                if not tail.isdigit():
                    continue
                metrics = load_json(metrics_path)
                observed_samples = int(metrics.get("num_samples", num_samples))
                if observed_samples != num_samples:
                    raise ValueError(
                        f"{metrics_path}: expected {num_samples} samples, "
                        f"found {observed_samples}"
                    )
                step = int(tail)
                rows.append(
                    {
                        "task": "line_tracing",
                        "seed": seed,
                        "method": method,
                        "step": step,
                        "cumulative_flops": LINE_RATE[method] * step * accum * NUM_PROCESSES,
                        "headline": float(metrics["max_correct_prefix_steps.mean"]),
                        "num_samples": observed_samples,
                        "num_valid": observed_samples,
                        "num_error": 0,
                        "num_unknown": 0,
                        "source": str(metrics_path.relative_to(ROOT)),
                    }
                )
            rows.sort(key=lambda row: row["step"])
            observed_steps = {row["step"] for row in rows}
            if strict_steps:
                validate_steps(
                    f"line tracing seed {seed} {method}",
                    observed_steps,
                    LINE_TRAIN_DIRS[(seed, method)],
                )
            else:
                expected_steps = numeric_steps(LINE_TRAIN_DIRS[(seed, method)])
                missing = sorted(expected_steps - observed_steps)
                extra = sorted(observed_steps - expected_steps)
                if missing or extra:
                    print(
                        f"[warn] line tracing seed {seed} {method}: "
                        f"missing={missing}, extra={extra}"
                    )
            curves[(seed, method)] = rows
    return curves


def validate_longmul_periodic_config(train_dir: Path) -> None:
    args = load_json(train_dir / "train_args.json")
    expected = {
        "eval_num_inference_steps": 50,
        "eval_seed": 1000,
        "eval_max_samples": 100,
        "eval_judge_k": 2,
        "eval_judge_strict": True,
    }
    mismatches = {key: (args.get(key), value) for key, value in expected.items()
                  if args.get(key) != value}
    if mismatches:
        raise ValueError(f"{train_dir}: periodic evaluation protocol mismatch {mismatches}")


def summarize_verdicts(path: Path) -> dict[str, int | float]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != LONGMUL_NUM_SAMPLES:
        raise ValueError(
            f"{path}: expected {LONGMUL_NUM_SAMPLES} verdicts, found {len(lines)}"
        )
    # Judge records include long nested reasoning traces.  The first ``correct``
    # field on each line is the top-level verdict, so extracting it avoids JSON
    # decoding the unused traces for every plot refresh.
    statuses = []
    for line in lines:
        match = re.search(r'"correct"\s*:\s*"([^"]+)"', line)
        if match is None:
            raise ValueError(f"{path}: missing top-level correct verdict")
        statuses.append(match.group(1))
    unexpected = sorted({value for value in statuses if value not in
                         {"true", "false", "error", "unknown"}})
    if unexpected:
        raise ValueError(f"{path}: unexpected verdicts {unexpected}")
    num_correct = statuses.count("true")
    num_valid = num_correct + statuses.count("false")
    return {
        "headline": num_correct / LONGMUL_NUM_SAMPLES,
        "num_samples": LONGMUL_NUM_SAMPLES,
        "num_valid": num_valid,
        "num_error": statuses.count("error"),
        "num_unknown": statuses.count("unknown"),
        "judge_accuracy_valid_only": num_correct / num_valid if num_valid else 0.0,
    }


def load_longmul_curves() -> dict[tuple[int, str], list[dict]]:
    csv_fallback = (
        ROOT / "results" / "reference" / "synthetic" / "long_multiplication"
        / "multiseed" / "curves.csv"
    )
    if csv_fallback.exists() and not any(d.exists() for d in LONGMUL_TRAIN_DIRS.values()):
        import csv
        curves = {}
        with open(csv_fallback, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                seed = int(r["seed"])
                method = r["method"]
                r["step"] = int(r["step"])
                r["cumulative_flops"] = float(r["cumulative_flops"])
                r["headline"] = float(r["headline"])
                curves.setdefault((seed, method), []).append(r)
        return curves

    curves = {}
    for seed in SEEDS:
        for method in METHODS:
            train_dir = LONGMUL_TRAIN_DIRS[(seed, method)]
            validate_longmul_periodic_config(train_dir)
            accum = grad_accum(train_dir, expected=2)
            rows = []
            for verdict_path in train_dir.glob("eval_periodic/step-*/judge_results.jsonl"):
                tail = verdict_path.parent.name.split("-", 1)[1]
                if not tail.isdigit():
                    continue
                step = int(tail)
                if method == "vanilla" and step > LONGMUL_VANILLA_MAX_STEPS:
                    continue
                summary = summarize_verdicts(verdict_path)
                rows.append(
                    {
                        "task": "long_multiplication",
                        "seed": seed,
                        "method": method,
                        "step": step,
                        "cumulative_flops": LONGMUL_RATE[method] * step * accum * NUM_PROCESSES,
                        **summary,
                        "source": str(verdict_path.relative_to(ROOT)),
                    }
                )
            rows.sort(key=lambda row: row["step"])
            if not rows:
                raise ValueError(f"{train_dir}: no periodic long-multiplication evaluations")
            curves[(seed, method)] = rows
    return curves


def paired_anchor(curves: dict[tuple[int, str], list[dict]], seed: int,
                  budget: float) -> dict:
    vanilla = [row for row in curves[(seed, "vanilla")]
               if row["cumulative_flops"] <= budget + 1.0]
    selfcond = [row for row in curves[(seed, "selfcond")]
                if row["cumulative_flops"] <= budget + 1.0]
    sc = max(selfcond, key=lambda row: row["cumulative_flops"])
    va = min(
        (row for row in vanilla if row["cumulative_flops"] >= sc["cumulative_flops"]),
        key=lambda row: row["cumulative_flops"],
    )
    return {
        "vanilla_step": va["step"],
        "selfcond_step": sc["step"],
        "vanilla_flops": va["cumulative_flops"],
        "selfcond_flops": sc["cumulative_flops"],
        "vanilla": va["headline"],
        "selfcond": sc["headline"],
        "difference": sc["headline"] - va["headline"],
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def interpolate_curves(curves: dict[tuple[int, str], list[dict]], method: str,
                       budget: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = []
    for seed in SEEDS:
        rows = [row for row in curves[(seed, method)]
                if row["cumulative_flops"] <= budget + 1.0]
        xs = np.asarray([row["cumulative_flops"] / 1e18 for row in rows])
        ys = np.asarray([row["headline"] * scale for row in rows])
        series.append((xs, ys))
    lower = max(xs[0] for xs, _ in series)
    upper = min(xs[-1] for xs, _ in series)
    grid = np.asarray(sorted({float(x) for xs, _ in series for x in xs
                              if lower <= x <= upper}))
    values = np.stack([np.interp(grid, xs, ys) for xs, ys in series])
    return grid, values.mean(axis=0), values.std(axis=0, ddof=1)


def write_outputs(task_slug: str, suffix: str,
                  curves: dict[tuple[int, str], list[dict]], budget: float,
                  protocol: str,
                  schedule: str = "linear (mu=0, no dynamic shift)",
                  ) -> tuple[Path, Path, dict]:
    out_dir = ROOT / "outputs" / task_slug / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"curves_{suffix}.csv"
    fields = [
        "task", "seed", "method", "step", "cumulative_flops", "headline",
        "num_samples", "num_valid", "num_error", "num_unknown", "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for seed in SEEDS:
            for method in METHODS:
                writer.writerows(curves[(seed, method)])

    anchors = {str(seed): paired_anchor(curves, seed, budget) for seed in SEEDS}
    stats = {}
    for method in METHODS:
        values = [anchors[str(seed)][method] for seed in SEEDS]
        mean, sd = mean_sd(values)
        stats[method] = {
            "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
            "mean": mean,
            "sample_sd": sd,
            "n": len(values),
        }
    differences = [anchors[str(seed)]["difference"] for seed in SEEDS]
    diff_mean, diff_sd = mean_sd(differences)
    stats["paired_difference_selfcond_minus_vanilla"] = {
        "per_seed": {str(seed): value for seed, value in zip(SEEDS, differences)},
        "mean": diff_mean,
        "sample_sd": diff_sd,
        "n": len(differences),
    }
    summary = {
        "task": task_slug,
        "schedule": schedule,
        "seeds": list(SEEDS),
        "protocol": protocol,
        "vanilla_budget_flops": budget,
        "anchors": anchors,
        "statistics": stats,
    }
    summary_path = out_dir / f"summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path, summary


def plot_task(task_slug: str, preview_suffix: str,
              curves: dict[tuple[int, str], list[dict]], budget: float,
              summary: dict, *, scale: float, ylabel: str, title: str,
              ylim: tuple[float, float], paper_name: str,
              annotation_offsets: dict[str, tuple[float, float]] | None = None,
              legend_anchor: tuple[float, float] = (0.16, 0.50),
              annotation_colors: dict[str, str] | None = None) -> tuple[Path, Path]:
    set_style()
    fig, ax = plt.subplots(figsize=(4.3, 2.75))
    style_axis(ax)
    for method in METHODS:
        xs, means, sds = interpolate_curves(curves, method, budget, scale)
        ax.fill_between(xs, means - sds, means + sds, color=COLORS[method],
                        alpha=0.16, linewidth=0, zorder=1)
        ax.plot(xs, means, color=COLORS[method], linewidth=CURVE_WIDTH,
                solid_capstyle="round", zorder=3)

    ax.axvline(budget / 1e18, color=BUDGET_COLOR, linestyle=(0, (3, 2)),
               linewidth=BUDGET_WIDTH, zorder=0)
    offsets = annotation_offsets or {"vanilla": (-12, 8), "selfcond": (-12, -10)}
    for method in METHODS:
        offset = offsets[method]
        text_color = (annotation_colors or COLORS)[method]
        stats = summary["statistics"][method]
        anchors = [summary["anchors"][str(seed)] for seed in SEEDS]
        x = statistics.mean(anchor[f"{method}_flops"] for anchor in anchors) / 1e18
        y = stats["mean"] * scale
        ax.scatter([x], [y], s=10, color=COLORS[method], zorder=4)
        ax.annotate(
            f"{y:.1f} ± {stats['sample_sd'] * scale:.1f}",
            xy=(x, y), xytext=offset, textcoords="offset points",
            ha="right", va="bottom" if offset[1] > 0 else "top",
            fontsize=7.0, color=text_color,
        )

    budget_x = budget / 1e18
    ax.set_xlim(-0.025 * budget_x, budget_x * 1.035)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, labelpad=2.2)
    ax.set_xlabel(r"Cumulative FLOPs ($\times 10^{18}$)", labelpad=4.5)
    ax.set_title(title, loc="left", pad=6)
    handles = [Line2D([0], [0], color=COLORS[method], linewidth=CURVE_WIDTH,
                      label=LABELS[method]) for method in METHODS]
    legend = ax.legend(handles=handles, loc="center left", bbox_to_anchor=legend_anchor,
                       frameon=True, fancybox=True, framealpha=1.0,
                       facecolor=NFE_BACKGROUND_COLOR, edgecolor=NFE_LEGEND_EDGE_COLOR,
                       borderpad=0.36, handlelength=1.75,
                       handletextpad=0.45, labelspacing=0.42)
    legend.get_frame().set_linewidth(0.45)
    legend.get_frame().set_boxstyle("round,pad=0.18,rounding_size=0.14")
    fig.tight_layout(pad=0.4)

    paper_path = FIGURE_ROOT / paper_name
    preview_path = ROOT / "outputs" / task_slug / "multiseed" / preview_suffix
    fig.savefig(paper_path, metadata={"Creator": "Matplotlib"})
    fig.savefig(preview_path, dpi=300)
    plt.close(fig)
    return paper_path, preview_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "line-tracing", "long-multiplication"),
        default="all",
        help="Task to plot (default: all).",
    )
    parser.add_argument(
        "--line-schedule",
        choices=("linear", "default"),
        default="linear",
        help="Line-tracing inference schedule (default: linear).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    output_paths = []

    if args.task in ("all", "line-tracing"):
        is_default = args.line_schedule == "default"
        line_eval_dirs = LINE_DEFAULT_EVAL_DIRS if is_default else LINE_EVAL_DIRS
        line_num_samples = 20 if is_default else LINE_NUM_SAMPLES
        line_suffix = (
            "default_nfe100_n20_3seeds"
            if is_default else "linear_nfe50_n200_3seeds"
        )
        line_schedule = (
            "default (original shifted schedule)"
            if is_default else "linear (mu=0, no dynamic shift)"
        )
        line_curves = load_line_curves(
            line_eval_dirs,
            num_samples=line_num_samples,
            strict_steps=not is_default,
        )
        line_budget = min(
            max(row["cumulative_flops"] for row in line_curves[(seed, "vanilla")])
            for seed in SEEDS
        )
        line_csv, line_summary_path, line_summary = write_outputs(
            "line-tracing",
            line_suffix,
            line_curves,
            line_budget,
            (
                "100 NFE, evaluation seed 1000, first 20 validation samples; "
                "default shifted schedule; CTD; SelfCondFM truncated to the "
                "common VanillaFM training-FLOP budget"
                if is_default else
                "50 NFE, evaluation seed 1000, first 200 validation samples; "
                "CTD; SelfCondFM truncated to the common VanillaFM "
                "training-FLOP budget"
            ),
            schedule=line_schedule,
        )
        line_pdf, line_png = plot_task(
            "line-tracing", f"eval_vs_flops_multiseed_{line_suffix}.png",
            line_curves, line_budget, line_summary, scale=1.0,
            ylabel="Longest correct prefix", title="" if is_default else "Line tracing", ylim=(-2, 56),
            paper_name=(
                "multi-seed-line-tracing-default-schedule.pdf"
                if is_default else "multi-seed-line-tracing.pdf"
            ),
            annotation_offsets=(
                {"vanilla": (-3, 11), "selfcond": (0, -10)}
                if is_default else None
            ),
            legend_anchor=(0.66, 0.40) if is_default else (0.16, 0.50),
            annotation_colors=None,
        )
        output_paths.extend((line_csv, line_summary_path, line_pdf, line_png))

    if args.task in ("all", "long-multiplication"):
        longmul_curves = load_longmul_curves()
        longmul_budget = min(
            max(row["cumulative_flops"] for row in longmul_curves[(seed, "vanilla")])
            for seed in SEEDS
        )
        longmul_csv, longmul_summary_path, longmul_summary = write_outputs(
            "long-multiplication",
            "default_nfe50_n100_3seeds",
            longmul_curves,
            longmul_budget,
            "50 NFE, evaluation seed 1000, 100-example test set, strict k=2 judge; "
            "normal schedule periodic evaluation; pass rate is true verdicts / 100; "
            "VanillaFM restricted to 125k steps",
        )
        longmul_pdf, longmul_png = plot_task(
            "long-multiplication", "eval_vs_flops_multiseed_default_nfe50_n100.png",
            longmul_curves, longmul_budget, longmul_summary, scale=100.0,
            ylabel="Accuracy (%)", title="", ylim=(-2, 105),
            paper_name="multi-seed-long-multiplication.pdf",
            annotation_offsets={"vanilla": (-2, 6), "selfcond": (4, -8)},
            legend_anchor=(0.09, 0.64),
            annotation_colors=None,
        )
        output_paths.extend((longmul_csv, longmul_summary_path, longmul_pdf, longmul_png))

    for path in output_paths:
        print(f"[out] {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
