#!/usr/bin/env python3
"""Render the long-text-rendering CER and NFE inference-FLOPs figures as vector PDFs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
RESULTS_REFERENCE = ROOT / "results" / "reference"
FIGURE_ROOT = ROOT / "outputs" / "figures"
CER_JSON = RESULTS_REFERENCE / "synthetic" / "long_text_rendering" / "cer_curves.json"
NFE_RESULT_ROOT = ROOT / "outputs" / "line_tracing" / "nfe-scaling"
INFERENCE_FLOPS_JSON = (
    RESULTS_REFERENCE / "nfe_scaling" / "line_tracing" / "inference_flops.json"
)

WIDTH, HEIGHT = 3.55 * 72, 2.20 * 72
LEFT, RIGHT, BOTTOM, TOP = 35.0, 5.5, 27.0, 7.0
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - BOTTOM - TOP

BLUE = "#5572A9"
ORANGE = "#CE885F"
GRID = "#FFFFFF"
TEXT = "#000000"
SPINE = "#555555"
LEGEND_FRAME = "#C8C8C8"
PANEL = "#F1F1F6"
LEGEND_FRAME_INSET = 6.0


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _set_color(pdf: canvas.Canvas, color: str, *, stroke: bool = True) -> None:
    color = color.lstrip("#")
    rgb = tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    if stroke:
        pdf.setStrokeColorRGB(*rgb)
    else:
        pdf.setFillColorRGB(*rgb)


def _draw_legend_frame(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill_color: str = PANEL,
) -> None:
    """Rounded legend box filled with the plot's background color."""
    _set_color(pdf, fill_color, stroke=False)
    _set_color(pdf, LEGEND_FRAME)
    pdf.setLineWidth(0.45)
    pdf.roundRect(x, y, width, height, 2.0, stroke=1, fill=1)


def _draw_axes(
    pdf: canvas.Canvas,
    *,
    x_ticks: list[tuple[float, str]],
    y_ticks: list[tuple[float, str]],
    x_label: str,
    y_label: str,
    y_minor: list[float] | None = None,
    template: bool = False,
) -> None:
    if template:
        pdf.setFillColorRGB(*tuple(int(PANEL[i : i + 2], 16) / 255 for i in (1, 3, 5)))
        pdf.rect(LEFT, BOTTOM, PLOT_W, PLOT_H, stroke=0, fill=1)
    _set_color(pdf, GRID)
    pdf.setLineWidth(0.45)
    for y, _ in y_ticks:
        py = BOTTOM + y * PLOT_H
        pdf.line(LEFT, py, LEFT + PLOT_W, py)
    if template:
        for x, _ in x_ticks:
            px = LEFT + x * PLOT_W
            pdf.line(px, BOTTOM, px, BOTTOM + PLOT_H)
    if y_minor:
        pdf.setLineWidth(0.30)
        for y in y_minor:
            py = BOTTOM + y * PLOT_H
            pdf.line(LEFT, py, LEFT + PLOT_W, py)

    if not template:
        _set_color(pdf, SPINE)
        pdf.setLineWidth(0.60)
        pdf.line(LEFT, BOTTOM, LEFT, BOTTOM + PLOT_H)
        pdf.line(LEFT, BOTTOM, LEFT + PLOT_W, BOTTOM)

    _set_color(pdf, TEXT, stroke=False)
    pdf.setFont("Helvetica", 6.5)
    for x, label in x_ticks:
        px = LEFT + x * PLOT_W
        if not template:
            pdf.setStrokeColorRGB(0.33, 0.33, 0.33)
            pdf.setLineWidth(0.50)
            pdf.line(px, BOTTOM, px, BOTTOM - 2.6)
        _set_color(pdf, TEXT, stroke=False)
        pdf.drawCentredString(px, BOTTOM - 10.0, label)
    for y, label in y_ticks:
        py = BOTTOM + y * PLOT_H
        if not template:
            pdf.setStrokeColorRGB(0.33, 0.33, 0.33)
            pdf.setLineWidth(0.50)
            pdf.line(LEFT - 2.6, py, LEFT, py)
        _set_color(pdf, TEXT, stroke=False)
        pdf.drawRightString(LEFT - 5.0, py - 2.2, label)

    pdf.setFont("Helvetica", 7.2)
    pdf.drawCentredString(LEFT + PLOT_W / 2, 5.0, x_label)
    pdf.saveState()
    pdf.translate(8.0, BOTTOM + PLOT_H / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, y_label)
    pdf.restoreState()


def _draw_polyline(
    pdf: canvas.Canvas,
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float = 1.05,
) -> None:
    _set_color(pdf, color)
    pdf.setLineWidth(width)
    pdf.setLineJoin(1)
    pdf.setLineCap(1)
    path = pdf.beginPath()
    for index, (x, y) in enumerate(points):
        px = LEFT + x * PLOT_W
        py = BOTTOM + y * PLOT_H
        if index == 0:
            path.moveTo(px, py)
        else:
            path.lineTo(px, py)
    pdf.drawPath(path)


CER_SERIES = (
    ("vanilla", "VanillaFM", BLUE),
    ("selfcond", "SelfCondFM", ORANGE),
)


def _log_position(value: float, ymin: float, ymax: float) -> float:
    lo, hi = math.log10(ymin), math.log10(ymax)
    return (math.log10(value) - lo) / (hi - lo)


def plot_long_text_ocr(source: Path, output: Path) -> None:
    data = _load_json(source)
    xmax = 50.0  # cumulative FLOPs in units of 1e18
    ymin, ymax = 0.003, 0.4

    pdf = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    pdf.setTitle("Character error rate of long-text rendering against training FLOPs")
    pdf.setCreator("plot_supporting_analyses.py")
    _draw_axes(
        pdf,
        x_ticks=[(0.03 + 0.94 * v / xmax, str(v)) for v in (0, 10, 20, 30, 40, 50)],
        y_ticks=[(0.03 + 0.94 * _log_position(v, ymin, ymax), f"{v:g}") for v in (0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)],
        y_minor=[
            0.03 + 0.94 * _log_position(v, ymin, ymax)
            for v in (0.004, 0.006, 0.007, 0.008, 0.009, 0.03, 0.04, 0.06, 0.07, 0.08, 0.09, 0.3)
        ],
        x_label="",
        y_label="Character error rate",
        template=True,
    )

    for key, _, color in CER_SERIES:
        points = [
            (
                0.03 + 0.94 * float(h["flops"]) / 1e18 / xmax,
                0.03 + 0.94 * _log_position(float(h["cer_mean"]), ymin, ymax),
            )
            for h in data["series"][key]
        ]
        _draw_polyline(pdf, points, color=color, width=0.95)

    # "Cumulative FLOPs (x10^18)" with a raised exponent.
    pdf.setFont("Helvetica", 7.2)
    part1 = "Cumulative FLOPs (×10"
    part2 = "18"
    part3 = ")"
    w1 = pdf.stringWidth(part1, "Helvetica", 7.2)
    w2 = pdf.stringWidth(part2, "Helvetica", 5.2)
    w3 = pdf.stringWidth(part3, "Helvetica", 7.2)
    x0 = LEFT + (PLOT_W - w1 - w2 - w3) / 2
    _set_color(pdf, TEXT, stroke=False)
    pdf.drawString(x0, 5.0, part1)
    pdf.setFont("Helvetica", 5.2)
    pdf.drawString(x0 + w1, 7.0, part2)
    pdf.setFont("Helvetica", 7.2)
    pdf.drawString(x0 + w1 + w2, 5.0, part3)

    _draw_legend_vertical(
        pdf,
        [(label, color) for _, label, color in CER_SERIES],
        BOTTOM + PLOT_H - LEGEND_FRAME_INSET - 4.5,
    )

    pdf.showPage()
    pdf.save()


# Legend anchor (fraction of the plot area) for the inference-FLOPs figure.
TRAJECTORY_LEGEND_Y = 0.52


def _draw_legend_vertical(
    pdf: canvas.Canvas,
    entries: list[tuple[str, str]],
    y_top: float,
    *,
    right_inset: float = LEGEND_FRAME_INSET,
    dy: float = 8.0,
    fill_color: str = PANEL,
) -> None:
    """Right-aligned vertical legend with space inside the plot boundary."""
    pdf.setFont("Helvetica", 6.0)
    item_width = max(
        12 + pdf.stringWidth(label, "Helvetica", 6.0) for label, _ in entries
    )
    # The frame extends 3 pt beyond the swatches and labels on each side.
    x = LEFT + PLOT_W - item_width - right_inset - 3.0
    box_height = (len(entries) - 1) * dy + 9.5
    _draw_legend_frame(
        pdf,
        x - 3.0,
        y_top - (len(entries) - 1) * dy - 5.0,
        item_width + 6.0,
        box_height,
        fill_color=fill_color,
    )
    for label, color in entries:
        _set_color(pdf, color)
        pdf.setLineWidth(1.05)
        pdf.line(x, y_top, x + 9, y_top)
        _set_color(pdf, TEXT, stroke=False)
        pdf.drawString(x + 12, y_top - 2.0, label)
        y_top -= dy


INFERENCE_SERIES = (
    ("vanilla", "VanillaFM", BLUE),
    ("selfcond", "SelfCondFM", ORANGE),
)


def _draw_x_label_with_exponent(pdf: canvas.Canvas, text: str, exponent: str) -> None:
    """Center an x-axis label whose unit carries a raised exponent."""
    pdf.setFont("Helvetica", 7.2)
    exponent_size = 5.2
    width = pdf.stringWidth(text, "Helvetica", 7.2) + pdf.stringWidth(
        exponent, "Helvetica", exponent_size
    ) + pdf.stringWidth(")", "Helvetica", 7.2)
    x0 = LEFT + (PLOT_W - width) / 2
    _set_color(pdf, TEXT, stroke=False)
    pdf.drawString(x0, 5.0, text)
    pdf.setFont("Helvetica", exponent_size)
    pdf.drawString(x0 + pdf.stringWidth(text, "Helvetica", 7.2), 7.0, exponent)
    pdf.setFont("Helvetica", 7.2)
    pdf.drawString(
        x0
        + pdf.stringWidth(text, "Helvetica", 7.2)
        + pdf.stringWidth(exponent, "Helvetica", exponent_size),
        5.0,
        ")",
    )


def plot_inference_flops(source: Path, output: Path) -> None:
    data = _load_json(source)
    flops = {
        method: {
            int(row["nfe"]): float(row[f"{method}_flops_per_sample"])
            for row in data["table"]
        }
        for method in ("vanilla", "selfcond")
    }
    xmax = 4.6  # cumulative inference FLOPs in units of 1e12
    ymax = 54.0

    pdf = canvas.Canvas(str(output), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    pdf.setTitle("Correct trace depth against cumulative inference FLOPs")
    pdf.setCreator("plot_supporting_analyses.py")
    _draw_axes(
        pdf,
        x_ticks=[(0.03 + 0.94 * v / xmax, str(v)) for v in (0, 1, 2, 3, 4)],
        y_ticks=[(0.03 + 0.94 * v / ymax, str(v)) for v in (0, 10, 20, 30, 40, 50)],
        x_label="",
        y_label="Longest correct prefix",
        template=True,
    )

    for method, _, color in INFERENCE_SERIES:
        points = []
        for nfe in sorted(flops[method]):
            metrics_path = (
                NFE_RESULT_ROOT / method / f"nfe_{nfe:03d}" / "seed_0000" / "metrics.json"
            )
            if not metrics_path.is_file():
                raise FileNotFoundError(f"missing NFE metrics: {metrics_path}")
            ctd = float(_load_json(metrics_path)["max_correct_prefix_steps.mean"])
            points.append(
                (
                    0.03 + 0.94 * flops[method][nfe] / 1e12 / xmax,
                    0.03 + 0.94 * ctd / ymax,
                )
            )
        _draw_polyline(pdf, points, color=color, width=0.95)

    _draw_x_label_with_exponent(pdf, "Inference FLOPs (×10", "12")

    _draw_legend_vertical(
        pdf,
        [(label, color) for _, label, color in INFERENCE_SERIES],
        BOTTOM + TRAJECTORY_LEGEND_Y * PLOT_H,
    )

    pdf.showPage()
    pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cer-json", type=Path, default=CER_JSON)
    parser.add_argument(
        "--inference-flops-json", type=Path, default=INFERENCE_FLOPS_JSON
    )
    args = parser.parse_args()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plot_long_text_ocr(args.cer_json, FIGURE_ROOT / "long-text-ocr.pdf")
    plot_inference_flops(
        args.inference_flops_json, FIGURE_ROOT / "inference-flops.pdf"
    )


if __name__ == "__main__":
    main()
