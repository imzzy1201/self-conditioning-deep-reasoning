#!/usr/bin/env python3
"""Render the paper's theory-aligned task and NFE figures from saved results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
from matplotlib.font_manager import findfont
from matplotlib.ft2font import FT2Font
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox


ROOT = Path(__file__).resolve().parents[2]
RESULTS_REFERENCE = ROOT / "results" / "reference"
# Frozen per-task curves live under results/reference/synthetic/<task>/.
CURVE_ROOT = RESULTS_REFERENCE / "synthetic"
OUTPUT_ROOT = ROOT / "outputs" / "04_nfe_scaling" / "line_tracing" / "eval"
FIGURE_ROOT = ROOT / "outputs" / "figures"
DEMO_ROOT = RESULTS_REFERENCE / "synthetic" / "theory_aligned_tasks" / "demo"

METHODS = ("vanilla", "selfcond")
LABELS = {"vanilla": "VanillaFM", "selfcond": "SelfCondFM"}
COLORS = {"vanilla": "#5572A9", "selfcond": "#CE885F"}
MARKERS = {"vanilla": "o", "selfcond": "s"}
BUDGET_COLOR = "#A7A7A7"
GRID_COLOR = "#E3E3E3"
SECTION_COLOR = "#1B4B7C"
SEPARATOR_COLOR = "#D8D8D8"
EXAMPLE_LABEL_X = -0.225
BASE_FONT_SIZE = 8.0
AXIS_LABEL_SIZE = 7.4
TICK_LABEL_SIZE = 7.2
LEGEND_SIZE = 7.4
THEORY_LINE_SCALE = 0.88
THEORY_CURVE_WIDTH = 0.80
THEORY_BUDGET_WIDTH = 0.65
THEORY_SEPARATOR_WIDTH = 0.70
# Target sizes after LaTeX scales each PDF to its final width.
FINAL_PANEL_TITLE_SIZE = 7.8
FINAL_SECTION_LABEL_SIZE = 6.6
FINAL_AXIS_LABEL_SIZE = 6.2
FINAL_TICK_LABEL_SIZE = 5.7
FINAL_LEGEND_SIZE = 6.2
THEORY_INCLUDE_SCALE = 0.95
# The NFE source is 0.45 text widths wide and is included at 0.9 * 0.4 text widths.
NFE_INCLUDE_SCALE = 0.9 * 0.4 / 0.45
NFE_BACKGROUND_COLOR = "#F1F1F6"
NFE_GRID_COLOR = "#FFFFFF"
NFE_TEXT_COLOR = "#000000"
NFE_LEGEND_EDGE_COLOR = "#DEDEDE"
NFE_COLORS = {"vanilla": "#5572A9", "selfcond": "#CE885F"}


def source_size(final_size: float, include_scale: float) -> float:
    """Convert a desired final LaTeX size to the corresponding source-PDF size."""
    return final_size / include_scale


def set_style() -> None:
    """Use compact sans-serif typography throughout the paper figures."""
    sns.set_theme(
        context="paper",
        style="ticks",
        font="Arial",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": 8.6,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "text.color": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titlecolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "savefig.facecolor": "white",
        },
    )


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def method_handles(
    *,
    include_budget: bool = False,
    markers: bool = True,
    line_width: float = 1.05,
    budget_line_width: float = 0.9,
    marker_size: float = 3.2,
    colors: dict[str, str] | None = None,
) -> list[Line2D]:
    colors = COLORS if colors is None else colors
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[method],
            marker=MARKERS[method] if markers else None,
            markersize=marker_size,
            markeredgewidth=0,
            linewidth=line_width,
            label=LABELS[method],
        )
        for method in METHODS
    ]
    if include_budget:
        handles.append(
            Line2D(
                [0],
                [0],
                color=BUDGET_COLOR,
                linestyle=(0, (3, 2)),
                linewidth=budget_line_width,
                label="Matched budget",
            )
        )
    return handles


def style_axis(ax: plt.Axes, *, line_scale: float = 1.0) -> None:
    """Apply the shared light-panel, white-grid figure template."""
    ax.set_axisbelow(True)
    ax.set_facecolor(NFE_BACKGROUND_COLOR)
    ax.grid(axis="both", color=NFE_GRID_COLOR, alpha=1.0,
            linewidth=0.55 * line_scale)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", which="both", length=0, width=0,
                   colors=NFE_TEXT_COLOR)


def setup_example_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_example_label(ax: plt.Axes, label: str, *, x: float, y: float) -> None:
    """Draw a compact section label with a small right-pointing marker."""
    ax.plot(
        x,
        y - 0.026,
        marker=">",
        markersize=2.9,
        color=SECTION_COLOR,
        markeredgewidth=0,
        clip_on=False,
    )
    ax.text(
        x + 0.032,
        y,
        label,
        ha="left",
        va="top",
        fontsize=source_size(FINAL_SECTION_LABEL_SIZE, THEORY_INCLUDE_SCALE),
        color=SECTION_COLOR,
    )


def place_demo_image(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
) -> tuple[float, float]:
    """Place an original demo image without changing its physical aspect ratio."""
    axis_box = ax.get_window_extent()
    axis_aspect = axis_box.width / axis_box.height
    image_aspect = image.shape[1] / image.shape[0]
    width = min(max_width, max_height * image_aspect / axis_aspect)
    height = min(max_height, max_width * axis_aspect / image_aspect)
    ax.imshow(
        image,
        extent=(x, x + width, y - height, y),
        interpolation="none",
        aspect="auto",
        zorder=1,
    )
    return width, height


def place_demo_image_at_scale(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    x: float,
    y: float,
    width_per_pixel: float,
) -> tuple[float, float]:
    """Place an original demo image at a fixed physical scale per source pixel."""
    axis_box = ax.get_window_extent()
    axis_aspect = axis_box.width / axis_box.height
    width = image.shape[1] * width_per_pixel
    height = image.shape[0] * width_per_pixel * axis_aspect
    ax.imshow(
        image,
        extent=(x, x + width, y - height, y),
        interpolation="none",
        aspect="auto",
        zorder=1,
    )
    return width, height


def draw_line_tracing_example(ax: plt.Axes) -> None:
    setup_example_axis(ax)
    draw_example_label(ax, "Input", x=EXAMPLE_LABEL_X, y=0.86)
    draw_example_label(ax, "Target", x=EXAMPLE_LABEL_X, y=0.47)
    input_image = mpimg.imread(DEMO_ROOT / "lt_i.png")
    target_image = mpimg.imread(DEMO_ROOT / "lt_o.png")
    image_x = 0.10
    input_top = 0.80
    target_top = 0.42
    input_width, input_height = place_demo_image(
        ax, input_image, x=image_x, y=input_top, max_width=0.80, max_height=0.27
    )
    target_width, target_height = place_demo_image(
        ax, target_image, x=image_x, y=target_top, max_width=0.80, max_height=0.27
    )
    ellipsis_gap = 0.035
    ax.text(image_x + input_width + ellipsis_gap, input_top - input_height / 2, r"$\cdots$", ha="left", va="center", fontsize=source_size(6.8, THEORY_INCLUDE_SCALE))
    ax.text(image_x + target_width + ellipsis_gap, target_top - target_height / 2, r"$\cdots$", ha="left", va="center", fontsize=source_size(6.8, THEORY_INCLUDE_SCALE))


def draw_multiplication_example(ax: plt.Axes) -> None:
    setup_example_axis(ax)
    draw_example_label(ax, "Input", x=EXAMPLE_LABEL_X, y=0.86)
    draw_example_label(ax, "Target", x=EXAMPLE_LABEL_X, y=0.635)
    # Use the same horizontal crop and pixel scale so the two operands align exactly.
    input_image = mpimg.imread(DEMO_ROOT / "mult_i.png")[15:48, 42:184]
    target_image = mpimg.imread(DEMO_ROOT / "mult_o.png")[15:180, 42:184]
    axis_box = ax.get_window_extent()
    axis_aspect = axis_box.width / axis_box.height
    width_per_pixel = min(
        0.73 / target_image.shape[1],
        0.60 / (target_image.shape[0] * axis_aspect),
    )
    place_demo_image_at_scale(
        ax,
        input_image,
        x=0.25,
        y=0.83,
        width_per_pixel=width_per_pixel,
    )
    place_demo_image_at_scale(
        ax,
        target_image,
        x=0.25,
        y=0.635,
        width_per_pixel=width_per_pixel,
    )


def draw_text_rendering_example(ax: plt.Axes) -> None:
    setup_example_axis(ax)
    draw_example_label(ax, "Input (text)", x=EXAMPLE_LABEL_X, y=0.86)

    ax.text(
        0.045,
        0.68,
        "Once upon a time, in a big forest, ...",
        family="sans-serif",
        fontsize=source_size(4.4, THEORY_INCLUDE_SCALE),
        ha="left",
        va="center",
    )
    ax.text(
        0.965,
        0.615,
        "... cold, and she wanted to climb it.",
        family="sans-serif",
        fontsize=source_size(4.4, THEORY_INCLUDE_SCALE),
        ha="right",
        va="center",
    )

    draw_example_label(ax, "Target", x=EXAMPLE_LABEL_X, y=0.51)
    target_image = mpimg.imread(DEMO_ROOT / "text_o.png")
    # Full-width paragraph (no horizontal crop); the block is stretched to span
    # exactly between the two dashed boundary lines below (x = 0.045 / 0.965).
    first_paragraph = target_image[10:123, 8:376]
    for boundary_x in (0.045, 0.965):
        ax.plot(
            (boundary_x, boundary_x),
            (0.095, 0.43),
            color=SECTION_COLOR,
            linewidth=0.55 * THEORY_LINE_SCALE,
            linestyle=(0, (2.0, 1.6)),
            solid_capstyle="butt",
        )
    place_demo_image(
        ax,
        first_paragraph,
        x=0.045,
        y=0.43,
        max_width=0.92,
        max_height=0.34,
    )


def plot_theory_aligned_tasks() -> Path:
    tasks = (
        {
            "path": (
                CURVE_ROOT
                / "line_tracing"
                / "curves_aligned_default_nfe100_n20.json"
            )
            if (
                CURVE_ROOT
                / "line_tracing"
                / "curves_aligned_default_nfe100_n20.json"
            ).exists()
            else (CURVE_ROOT / "line_tracing" / "curves_aligned.json"),
            "metric": "eval",
            "scale": 1.0,
            "ylabel": "Longest correct prefix",
            "ylim": (-1.0, 52.0),
            "yticks": range(0, 51, 10),
            "xticks": (0, 1, 2, 3, 4),
        },
        {
            "path": CURVE_ROOT / "long_multiplication" / "curves_aligned.json",
            "metric": "accuracy",
            "scale": 100.0,
            "ylabel": "Accuracy (%)",
            "ylim": (-2.0, 104.0),
            "yticks": range(0, 101, 20),
            "xticks": (0, 5, 10, 15, 20),
        },
        {
            "path": CURVE_ROOT / "long_text_rendering" / "curves_aligned.json",
            "metric": "num_correct_100",
            "scale": 100.0,
            "ylabel": "Accuracy (%)",
            "ylim": (-1.0, 52.0),
            "yticks": range(0, 51, 10),
            "xticks": (0, 10, 20, 30, 40),
        },
    )

    # Draw at the ICLR text width so LaTeX preserves the nominal font sizes.
    width = 5.5
    fig = plt.figure(figsize=(width, 2.90))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.066,
        right=0.995,
        bottom=0.275,
        top=0.985,
        wspace=0.36,
        hspace=0.035,
        height_ratios=(1.12, 0.82),
    )
    example_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]

    separator_xs = [left_ax.get_position().x1 + 0.012 for left_ax in axes[:-1]]
    for separator_x in separator_xs:
        fig.add_artist(
            Line2D(
                (separator_x, separator_x),
                (0.200, 0.985),
                transform=fig.transFigure,
                color=SEPARATOR_COLOR,
                linestyle=(0, (1.0, 2.0)),
                linewidth=THEORY_SEPARATOR_WIDTH,
                zorder=0,
            )
        )

    panel_edges = (0.0, *separator_xs, 1.0)
    titles = ("Line tracing", "Long multiplication", "Long text rendering")
    for title, left, right in zip(titles, panel_edges[:-1], panel_edges[1:]):
        fig.text(
            (left + right) / 2,
            0.992,
            title,
            ha="center",
            va="top",
            fontsize=source_size(FINAL_PANEL_TITLE_SIZE, THEORY_INCLUDE_SCALE),
            fontweight="bold",
            fontfamily="sans-serif",
        )

    draw_line_tracing_example(example_axes[0])
    draw_multiplication_example(example_axes[1])
    draw_text_rendering_example(example_axes[2])

    for ax, task in zip(axes, tasks):
        payload = load_json(task["path"])
        budget = float(payload["config"]["vanilla_max_flops"]) / 1e18

        for method in METHODS:
            rows = sorted(payload["runs"][method], key=lambda row: row["flops"])
            xs = [float(row["flops"]) / 1e18 for row in rows]
            ys = [float(row[task["metric"]]) * task["scale"] for row in rows]
            sns.lineplot(
                x=xs,
                y=ys,
                ax=ax,
                color=COLORS[method],
                marker=None,
                linewidth=THEORY_CURVE_WIDTH,
                markeredgecolor=COLORS[method],
                markeredgewidth=0,
                errorbar=None,
                sort=False,
                legend=False,
                solid_capstyle="round",
            )
            ax.lines[-1].set_markevery(max(1, len(xs) // 15))

        ax.axvline(
            budget,
            color=BUDGET_COLOR,
            linestyle=(0, (3, 2)),
            linewidth=THEORY_BUDGET_WIDTH,
            zorder=3,
        )
        ax.set_xlim(-0.025 * budget, budget * 1.035)
        ax.set_xticks(task["xticks"])
        ax.set_ylim(task["ylim"])
        ax.set_yticks(task["yticks"])
        ax.set_xlabel(
            r"FLOPs ($\times 10^{18}$)",
            fontsize=source_size(FINAL_TICK_LABEL_SIZE, THEORY_INCLUDE_SCALE),
            labelpad=3.0,
        )
        ax.set_ylabel(
            task["ylabel"],
            labelpad=2.2,
            fontsize=source_size(FINAL_TICK_LABEL_SIZE, THEORY_INCLUDE_SCALE),
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=source_size(FINAL_TICK_LABEL_SIZE * 0.90, THEORY_INCLUDE_SCALE),
        )
        style_axis(ax, line_scale=THEORY_LINE_SCALE)

    custom_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["vanilla"],
            linewidth=THEORY_CURVE_WIDTH,
            label="w/o self-conditioning",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["selfcond"],
            linewidth=THEORY_CURVE_WIDTH,
            label="w/ self-conditioning",
        ),
    ]
    legend = fig.legend(
        handles=custom_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.100),
        ncol=2,
        frameon=False,
        fancybox=True,
        framealpha=1.0,
        fontsize=source_size(FINAL_LEGEND_SIZE, THEORY_INCLUDE_SCALE),
        borderpad=0.38,
        borderaxespad=0.0,
        handlelength=1.90,
        handletextpad=0.55,
        columnspacing=1.85,
    )
    for text in legend.get_texts():
        text.set_color(NFE_TEXT_COLOR)
    output = FIGURE_ROOT / "theory-aligned-tasks.pdf"
    crop_bottom = 0.100 * fig.get_figheight()
    crop_box = Bbox.from_extents(
        0,
        crop_bottom,
        fig.get_figwidth(),
        fig.get_figheight(),
    )
    fig.savefig(
        output,
        bbox_inches=crop_box,
        pad_inches=0,
        metadata={"Creator": "Matplotlib + seaborn"},
    )
    fig.savefig(
        output.with_suffix(".png"),
        bbox_inches=crop_box,
        pad_inches=0,
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    return output


def load_nfe_results() -> dict[str, list[tuple[int, float]]]:
    results: dict[str, list[tuple[int, float]]] = {}
    for method in METHODS:
        values = []
        for nfe_dir in (OUTPUT_ROOT / method).glob("nfe_*"):
            metrics_path = nfe_dir / "seed_0000" / "metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = load_json(metrics_path)
            nfe = int(nfe_dir.name.removeprefix("nfe_"))
            values.append((nfe, float(metrics["max_correct_prefix_steps.mean"])))
        if not values:
            raise FileNotFoundError(f"No NFE results found for {method} under {OUTPUT_ROOT}")
        results[method] = sorted(values)
    return results


def plot_nfe_scaling() -> Path:
    results = load_nfe_results()
    # This figure is placed at 0.45\textwidth in the paper.
    width = 0.45 * 5.5
    height = 1.53
    top = 0.915
    bottom = 0.215
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(NFE_BACKGROUND_COLOR)

    for method in METHODS:
        xs, ys = zip(*results[method])
        sns.lineplot(
            x=xs,
            y=ys,
            ax=ax,
            color=NFE_COLORS[method],
            marker=None,
            linewidth=THEORY_CURVE_WIDTH * THEORY_INCLUDE_SCALE / NFE_INCLUDE_SCALE,
            errorbar=None,
            sort=False,
            legend=False,
            solid_capstyle="round",
        )

    # Leave space outside the zero gridlines, which form the visible left and
    # lower boundaries without drawing a surrounding axes frame.
    ax.set_xlim(-1.4, 51.0)
    ax.set_xticks((0, 10, 20, 30, 40, 50))
    ax.set_ylim(-2.2, 58.0)
    ax.set_yticks((0, 10, 20, 30, 40, 50))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    nfe_axis_label_size = source_size(FINAL_AXIS_LABEL_SIZE, NFE_INCLUDE_SCALE)
    nfe_tick_label_size = source_size(FINAL_TICK_LABEL_SIZE, NFE_INCLUDE_SCALE)
    nfe_legend_size = source_size(FINAL_LEGEND_SIZE, NFE_INCLUDE_SCALE)

    # Highlight operating points using values from the selected result cells.
    ref_markers = {
        "selfcond": (10, (3.0, -3.0), "left", "top"),
        "vanilla": (6, (3.0, -3.0), "left", "top"),
    }
    for method, (nfe, offset, ha, va) in ref_markers.items():
        if nfe not in dict(results[method]):
            continue
        value = dict(results[method])[nfe]
        ax.plot(
            nfe,
            value,
            marker="o",
            markersize=2.6,
            markeredgecolor="white",
            markeredgewidth=0.35,
            color=NFE_COLORS[method],
            linestyle="none",
            zorder=3,
        )
        ax.annotate(
            f"{value:.2f}",
            xy=(nfe, value),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=nfe_tick_label_size * 0.8,
            color=NFE_COLORS[method],
            zorder=4,
        )

    ax.set_xlabel("NFE", labelpad=2.2, fontsize=nfe_axis_label_size)
    ax.set_ylabel(
        "Longest correct prefix",
        labelpad=2.6,
        fontsize=nfe_axis_label_size,
        # Vertical anchor: midpoint of the 0 and 50 tick heights, i.e. data
        # y=25 expressed in axes fraction over the (-2.2, 58.0) ylim.
        y=27.2 / 60.2,
    )
    ax.xaxis.label.set_color(NFE_TEXT_COLOR)
    ax.yaxis.label.set_color(NFE_TEXT_COLOR)
    ax.set_axisbelow(True)
    ax.grid(
        visible=True,
        which="major",
        axis="both",
        color=NFE_GRID_COLOR,
        alpha=1.0,
        linewidth=0.58 * THEORY_INCLUDE_SCALE / NFE_INCLUDE_SCALE,
    )
    ax.tick_params(
        axis="both",
        which="both",
        length=0,
        width=0,
        colors=NFE_TEXT_COLOR,
        labelsize=nfe_tick_label_size,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(
        handles=method_handles(
            markers=False,
            line_width=THEORY_CURVE_WIDTH * THEORY_INCLUDE_SCALE / NFE_INCLUDE_SCALE,
            colors=NFE_COLORS,
        ),
        loc="center right",
        bbox_to_anchor=(0.96, 0.46),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor=NFE_BACKGROUND_COLOR,
        edgecolor=NFE_LEGEND_EDGE_COLOR,
        fontsize=nfe_legend_size * 0.90,
        borderpad=0.30,
        borderaxespad=0.0,
        handlelength=1.7,
        handletextpad=0.40,
        labelspacing=0.28,
    )
    for text in legend.get_texts():
        text.set_color(NFE_TEXT_COLOR)
    legend.get_frame().set_linewidth(0.45)
    legend.get_frame().set_boxstyle("round,pad=0.16,rounding_size=0.12")
    fig.subplots_adjust(left=0.158, right=0.985, bottom=bottom, top=top)

    # Keep a small margin around every visible artist without exporting the
    # unused canvas area around the plot and labels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    crop_box = fig.get_tightbbox(renderer).padded(0.025)
    # The x label's layout box reserves font descent below the "NFE" glyphs,
    # which reads as a blank strip under the label; lift the bottom crop up to
    # the glyph ink so the pad matches the other three sides.
    xlabel_box = ax.xaxis.label.get_window_extent(renderer)
    label_prop = ax.xaxis.label.get_fontproperties()
    label_font = FT2Font(findfont(label_prop))
    # FT2Font metrics are in font units; the layout box of the label reserves
    # this much space below the baseline even though "NFE" has no descenders.
    descent_in = (
        -label_font.descender / label_font.units_per_EM * label_prop.get_size() / 72.0
    )
    crop_box.y0 = xlabel_box.y0 / renderer.dpi + descent_in - 0.025
    output = FIGURE_ROOT / "nfe-scaling.pdf"
    fig.savefig(
        output,
        bbox_inches=crop_box,
        pad_inches=0,
        facecolor="white",
        metadata={"Creator": "Matplotlib + seaborn"},
    )
    fig.savefig(
        output.with_suffix(".png"),
        bbox_inches=crop_box,
        pad_inches=0,
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)
    return output


def main() -> None:
    global OUTPUT_ROOT, CURVE_ROOT, FIGURE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("theory", "nfe", "both"), default="both")
    parser.add_argument("--nfe_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--curve_root", type=Path, default=CURVE_ROOT,
                        help="Main curves default to frozen paper reference files.")
    parser.add_argument("--figure_root", type=Path, default=FIGURE_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT, CURVE_ROOT, FIGURE_ROOT = args.nfe_root, args.curve_root, args.figure_root
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    set_style()
    outputs = []
    if args.figure in ("theory", "both"):
        outputs.append(plot_theory_aligned_tasks())
    if args.figure in ("nfe", "both"):
        outputs.append(plot_nfe_scaling())
    for output in outputs:
        print(f"[out] {output}")


if __name__ == "__main__":
    main()
