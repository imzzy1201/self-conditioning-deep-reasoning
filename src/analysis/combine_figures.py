#!/usr/bin/env python
"""Horizontally merge the three per-task eval_vs_flops figures into one panel.

Concatenates the line-tracing, long-multiplication, and long-text-rendering
curves side by side and stamps a (a)/(b)/(c) panel label above each, matching
the paper's controlled-results panel order. Outputs
outputs/analysis/combined_eval_vs_flops.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PANELS = [
    ("(a) Line tracing", "line_tracing"),
    ("(b) Long multiplication", "long_multiplication"),
    ("(c) Long text rendering", "long_text_rendering"),
]
GAP = 24          # px between panels
TOP = 40          # px strip for the panel labels
OUT = ROOT / "outputs" / "analysis" / "combined_eval_vs_flops.png"


def main() -> None:
    imgs = [
        Image.open(ROOT / "outputs" / f / "eval_vs_flops.png") for _, f in PANELS
    ]
    w = sum(im.width for im in imgs) + GAP * (len(imgs) - 1)
    h = TOP + max(im.height for im in imgs)
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    x = 0
    for (label, _), im in zip(PANELS, imgs):
        draw.text((x + 12, 8), label, fill="black", font=font)
        canvas.paste(im, (x, TOP))
        x += im.width + GAP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"[out] combined -> {OUT} ({canvas.size})")


if __name__ == "__main__":
    main()