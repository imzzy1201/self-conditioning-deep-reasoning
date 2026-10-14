"""Partial-product (多位数×一位数) judge for column-multiplication images.

Scores, per generated image:
  - per-partial-product correctness (0/1)  → "多位数×一位数" accuracy.
    A sample has ``num_digits_b`` partial products P_i = a × b_digits[i], each placed
    on its own row by the dataset renderer. Per-image partial-product acc =
    mean of the num_digits_b 0/1 values (matches outline #3's spec).
  - total correctness (0/1)                  → does the final result row equal a*b.

Truncation is judged at the *cell* level: a partial product P_i occupies grid
columns ``[shift+i .. shift+i+num_digits_a]`` (the last col is the carry-overflow
slot — a digit iff the GT partial product overflows, blank otherwise). Each
expected cell is read off the generated image; a missing digit where one is
expected, or an extra digit where a blank is expected, makes that PP wrong.

Geometry is auto-calibrated from the *target* image size (the dataset stores no
config): num_cols = A+B+1, num_rows = 3B+1, and (cell_size, padding) are solved
from (W, H). The digit classifier defaults to deterministic template matching
(reusing the renderer's font/anchor logic); a VLM row-strip backend shares the
same cell-level scoring via ``classify_row`` (see ``--digit_classifier vlm`` in
the entry script).

Pure logic: PIL + numpy only, no torch/pipeline dependency, so it is unit-
testable on CPU and re-runnable on already-saved prediction PNGs.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# The column-multiplication dataset is a flat script dir (no __init__.py, flat
# `from arithmetic import ...` imports), so we put it on sys.path to reuse the
# EXACT ground-truth/arithmetic/layout the generator used — no re-implementation.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LM_DIR = os.path.normpath(os.path.join(_HERE, "..", "long_multiplication_dataset"))
if _LM_DIR not in sys.path:
    sys.path.insert(0, _LM_DIR)

from arithmetic import compute_multiplication, digit_to_char  # noqa: E402
from config import ColumnMulConfig, DEFAULT_FONT_NAME  # noqa: E402
from layout import create_layout  # noqa: E402
from renderer import _load_font  # noqa: E402  (reuse the generator's font loader)


# ───────────────────────── geometry calibration ──────────────────────────────


def calibrate_geometry(
    width: int, height: int, num_digits_a: int, num_digits_b: int
) -> tuple[int, int, int, int]:
    """Solve (cell_size, padding, num_cols, num_rows) from a target image's size.

    The renderer draws on a grid num_cols = A+B+1 wide, num_rows = 3B+1 tall,
    with raw_width = 2*pad + num_cols*cs, raw_height = 2*pad + num_rows*cs
    (then rounded up to a multiple of 16, which is a no-op for the default
    cs=pad=16). Two equations, two unknowns → exact solve when num_cols !=
    num_rows; the degenerate case falls back to the default pad==cs assumption.
    """
    num_cols = num_digits_a + num_digits_b + 1
    num_rows = 3 * num_digits_b + 1
    if num_cols != num_rows and (width - height) % (num_cols - num_rows) == 0:
        cs = (width - height) // (num_cols - num_rows)
        pad = (width - num_cols * cs) // 2
        if cs > 0 and pad >= 0 and 2 * pad + num_cols * cs <= width and 2 * pad + num_rows * cs <= height:
            return cs, pad, num_cols, num_rows
    # Degenerate / rounding-fudged fallback: assume pad == cs (the default config).
    cs = max(1, round(width / (num_cols + 2)))
    pad = (width - num_cols * cs) // 2
    if pad < 0:
        pad = 0
    return cs, pad, num_cols, num_rows


def pp_row(i: int) -> int:
    """Grid row of partial product P_i (mirrors layout.create_layout's bookkeeping).

    P_0 → row 3; P_i (i≥1) → row 3*i+1. Cross-validated by _selfcheck() against
    create_layout for A,B ∈ [1,6].
    """
    return 3 if i == 0 else 3 * i + 1


def result_row(num_digits_b: int) -> int:
    """Grid row of the final result (= last addition S row; = P_0 row when B=1).

    = 3*num_digits_b uniformly (B=1 → row 3 = P_0, since for single-digit b the
    partial product IS the result).
    """
    return 3 * num_digits_b


def cell_center(row: int, col: int, num_cols: int, cs: int, pad: int) -> tuple[float, float]:
    """Pixel center of grid cell (row, col). Mirrors renderer._cell_center:
    col 0 (ones) is rightmost; higher cols render leftward."""
    x = pad + (num_cols - 1 - col) * cs + cs / 2
    y = pad + row * cs + cs / 2
    return x, y


# ───────────────────────── expected (GT) cells ──────────────────────────────


@dataclass
class ExpectedCells:
    """Ground-truth cell layout for one sample, derived from the generator itself."""
    num_cols: int
    num_rows: int
    cs: int
    pad: int
    # per PP i: list of (col, expected_char_or_None) over span [shift+i .. shift+i+A]
    pp: list[list[tuple[int, str | None]]]
    # result row: list of (col, expected_char_or_None) over span [0 .. A+B]
    result: list[tuple[int, str | None]]


def build_expected(a: int, b: int, base: int, cs: int, pad: int) -> ExpectedCells:
    """Build expected per-cell chars by reusing compute_multiplication +
    create_layout (so the all-zero-collapse, leading-zero rendering, and carry
    placement match the generator exactly — no re-derivation).

    Carry cells (is_carry=True) are skipped: they annotate the carry into the
    next column and are NOT part of a partial product's value.
    """
    A = len(_digits(a, base))
    B = len(_digits(b, base))
    num_cols = A + B + 1
    num_rows = 3 * B + 1

    result = compute_multiplication(a, b, base)
    layout = create_layout(result, ColumnMulConfig(num_digits_a=A, num_digits_b=B, base=base), input_only=False)

    def _row_cells(row: int) -> dict[int, str]:
        # non-carry digit cells at this row, keyed by col
        out: dict[int, str] = {}
        for c in layout.cells:
            if c.row == row and not c.is_carry:
                out[c.col] = c.char
        return out

    pp_expected: list[list[tuple[int, str | None]]] = []
    for i in range(B):
        shift = i
        cells = _row_cells(pp_row(i))
        span_cols = list(range(shift, shift + A + 1))  # +1 = carry-overflow slot
        pp_expected.append([(col, cells.get(col)) for col in span_cols])

    rcells = _row_cells(result_row(B))
    # result can have up to A+B digits (+1 carry); read span [0 .. A+B] inclusive.
    result_span = list(range(0, A + B + 1))
    result_expected = [(col, rcells.get(col)) for col in result_span]

    return ExpectedCells(num_cols, num_rows, cs, pad, pp_expected, result_expected)


def _digits(n: int, base: int) -> list[int]:
    """Little-endian digit list (col 0 = ones). Mirrors arithmetic.get_digits."""
    if n == 0:
        return [0]
    out = []
    while n > 0:
        out.append(n % base)
        n //= base
    return out


# ───────────────────────── digit classifier (template matching) ─────────────


# Module-level defaults (the entry script may override these from CLI before
# judging). Kept here so judge_sample() can build a cs-matched classifier
# without the caller threading cs/base through.
FONT_NAME = DEFAULT_FONT_NAME  # must match the generator's default font
MATCH_THRESHOLD = 0.3
EMPTY_AREA_FRAC = 0.02
# If set (cs, pad), skip auto-calibration and use these geometry values directly.
# Lets the entry honor explicit --cell_size/--padding overrides for datasets
# whose image size doesn't cleanly solve the calibration (e.g. 'pad' mode).
GEOMETRY_OVERRIDE: tuple[int, int] | None = None
_CLASSIFIER_CACHE: dict[tuple[int, int, str, float, float], TemplateClassifier] = {}


def get_classifier(base: int, cs: int) -> TemplateClassifier:
    """Return a cached TemplateClassifier for (base, cs) + current module defaults.

    cs is constant across a bucket (geometry depends only on A,B + the fixed
    dataset config), so one classifier serves every sample in that bucket.
    """
    key = (base, cs, FONT_NAME, MATCH_THRESHOLD, EMPTY_AREA_FRAC)
    clf = _CLASSIFIER_CACHE.get(key)
    if clf is None:
        clf = TemplateClassifier(base, cs, FONT_NAME, MATCH_THRESHOLD, EMPTY_AREA_FRAC)
        _CLASSIFIER_CACHE[key] = clf
    return clf


class TemplateClassifier:
    """Deterministic per-cell digit classifier: binarize a centered crop and
    pick the renderer-font template with max IoU; an empty cell returns None.

    The crop side is 0.7*cs, centered on the cell — large enough to capture the
    digit glyph (height ~0.6*cs) yet excluding the carry annotation drawn at the
    cell's top-left corner (carry_cx = cx - cs/2, i.e. cs/2 left of center).
    """

    def __init__(self, base: int, cs: int, font_name: str = DEFAULT_FONT_NAME,
                 match_threshold: float = 0.3, empty_area_frac: float = 0.02):
        self.base = base
        self.cs = cs
        self.patch = max(1, int(round(0.7 * cs)))
        self.match_threshold = match_threshold
        self.empty_area_frac = empty_area_frac
        self._templates: dict[str, np.ndarray] = {}  # char -> bool foreground mask
        for d in range(base):
            ch = digit_to_char(d)
            self._templates[ch] = self._render_template(ch, cs, font_name)
        self._empty_area = self.patch * self.patch

    def _render_template(self, char: str, cs: int, font_name: str) -> np.ndarray:
        """Render a single digit centered in a cs×cs white cell (matching the
        renderer's font_size/anchor/positioning exactly), then take the central
        patch and binarize into a bool foreground mask."""
        img = Image.new("RGB", (cs, cs), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = _load_font(font_name, int(0.65 * 1.25 * cs))
        draw.text((cs / 2, cs / 2), char, font=font, fill=(0, 0, 0), anchor="mm")
        return self._patch_mask(img)

    def _patch_mask(self, img: Image.Image) -> np.ndarray:
        """Crop the central patch from img and return a bool (foreground) mask."""
        w, h = img.size
        cx, cy = w / 2, h / 2
        half = self.patch / 2
        box = (int(round(cx - half)), int(round(cy - half)),
               int(round(cx + half)), int(round(cy + half)))
        crop = img.convert("L").crop(box)
        arr = np.asarray(crop, dtype=np.int16)
        return arr < 128  # black-on-white: dark pixels are foreground

    def classify_patch(self, patch_mask: np.ndarray) -> str | None:
        fg = patch_mask
        area = int(fg.sum())
        if area < self.empty_area_frac * self._empty_area:
            return None  # blank cell (truncation / no carry overflow)
        best_ch, best_iou = None, -1.0
        t_area = fg.size
        for ch, tmask in self._templates.items():
            inter = int(np.logical_and(fg, tmask).sum())
            union = area + int(tmask.sum()) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_ch = iou, ch
        if best_iou < self.match_threshold:
            return None  # unrecognized garbage → treat as wrong (None ≠ any GT digit)
        return best_ch

    def classify_cell(self, image: Image.Image, row: int, col: int,
                      num_cols: int, cs: int, pad: int) -> str | None:
        """Crop the central patch of cell (row,col) from `image` and classify."""
        cx, cy = cell_center(row, col, num_cols, cs, pad)
        half = self.patch / 2
        box = (int(round(cx - half)), int(round(cy - half)),
               int(round(cx + half)), int(round(cy + half)))
        crop = image.convert("L").crop(box)
        arr = np.asarray(crop, dtype=np.int16)
        mask = arr < 128
        return self.classify_patch(mask)


# ───────────────────────── per-sample judging ───────────────────────────────


def judge_sample(
    pred_path: str,
    target_path: str,
    a: int,
    b: int,
    base: int,
    classifier: TemplateClassifier | None = None,
) -> dict[str, Any]:
    """Judge one predicted image against GT (a, b, base).

    Calibrates geometry from the target image, reads every expected PP cell and
    result cell off the prediction, and applies the cell-level truncation rule.
    Returns per-PP 0/1 list, per-image partial_product_acc, and total_correct.
    If `classifier` is None, a cs-matched one is built via get_classifier().
    """
    with Image.open(target_path) as ti:
        target = ti.convert("RGB")
    with Image.open(pred_path) as pi:
        pred = pi.convert("RGB")

    A = len(_digits(a, base))
    B = len(_digits(b, base))
    # Calibrate from the TARGET (ground-truth geometry). If the prediction was
    # produced at a different size (shouldn't happen — same canvas as the edit),
    # fall back to calibrating from the prediction. An explicit GEOMETRY_OVERRIDE
    # (set by the entry from --cell_size/--padding) skips calibration entirely.
    if GEOMETRY_OVERRIDE is not None:
        cs, pad = GEOMETRY_OVERRIDE
        num_cols = A + B + 1
        num_rows = 3 * B + 1
    else:
        cs, pad, num_cols, num_rows = calibrate_geometry(target.size[0], target.size[1], A, B)
        if pred.size != target.size:
            cs, pad, num_cols, num_rows = calibrate_geometry(pred.size[0], pred.size[1], A, B)

    if classifier is None:
        classifier = get_classifier(base, cs)
    expected = build_expected(a, b, base, cs, pad)

    # Partial products.
    pp_correct: list[int] = []
    pp_details: list[dict[str, Any]] = []
    for i, span in enumerate(expected.pp):
        row = pp_row(i)
        reads = [classifier.classify_cell(pred, row, col, num_cols, cs, pad) for col, _ in span]
        gt = [gt_char for _, gt_char in span]
        ok = (reads == gt)  # cell-level: each cell must match (digit or None/blank)
        pp_correct.append(1 if ok else 0)
        pp_details.append({"row": row, "shift": i, "gt": gt, "pred": reads, "correct": bool(ok)})

    # Final result.
    rrow = result_row(B)
    r_reads = [classifier.classify_cell(pred, rrow, col, num_cols, cs, pad) for col, _ in expected.result]
    r_gt = [gt_char for _, gt_char in expected.result]
    total_correct = 1 if (r_reads == r_gt) else 0

    return {
        "num_digits_a": A,
        "num_digits_b": B,
        "base": base,
        "a": a,
        "b": b,
        "result": a * b,
        "pp_correct": pp_correct,
        "partial_product_acc": (sum(pp_correct) / len(pp_correct)) if pp_correct else 0.0,
        "total_correct": total_correct,
        # debugging detail (cheap; drop if too verbose)
        "pp_details": pp_details,
        "result_detail": {"row": rrow, "gt": r_gt, "pred": r_reads, "correct": bool(total_correct)},
        "geometry": {"cell_size": cs, "padding": pad, "num_cols": num_cols, "num_rows": num_rows,
                     "target_size": list(target.size), "pred_size": list(pred.size)},
    }


# ───────────────────────── aggregation ───────────────────────────────────────


def aggregate(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of judge_sample() results into bucket metrics.

    partial_product_acc: mean over samples of each sample's per-image PP acc
    (= mean of its num_digits_b 0/1 values), i.e. the outline's spec.
    total_acc: fraction of samples with total_correct==1.
    per_position_acc: accuracy broken down by PP position i (0-indexed).
    """
    n = len(per_sample)
    if n == 0:
        return {"n": 0, "partial_product_acc": None, "total_acc": None, "per_position_acc": {}}
    pp_accs = [s["partial_product_acc"] for s in per_sample]
    totals = [s["total_correct"] for s in per_sample]
    # per-position: collect correct[0/1] per PP index across samples that have it.
    by_pos: dict[int, list[int]] = {}
    for s in per_sample:
        for i, c in enumerate(s["pp_correct"]):
            by_pos.setdefault(i, []).append(c)
    per_position = {str(i): {"n": len(v), "acc": sum(v) / len(v)} for i, v in sorted(by_pos.items())}
    return {
        "n": n,
        "partial_product_acc": sum(pp_accs) / n,
        "total_acc": sum(totals) / n,
        "per_position_acc": per_position,
    }


# ───────────────────────── inspect overlay ───────────────────────────────────


def inspect_grid(target_path: str, a: int, b: int, base: int, out_path: str) -> str:
    """Overlay the calibrated grid + PP/result row highlights on the target image,
    saved to out_path. Visual sanity check that geometry + row bookkeeping align."""
    with Image.open(target_path) as ti:
        img = ti.convert("RGB").copy()
    A = len(_digits(a, base))
    B = len(_digits(b, base))
    cs, pad, num_cols, num_rows = calibrate_geometry(img.size[0], img.size[1], A, B)
    draw = ImageDraw.Draw(img, "RGBA")
    # grid cell borders
    for r in range(num_rows):
        for c in range(num_cols):
            cx, cy = cell_center(r, c, num_cols, cs, pad)
            half = cs / 2
            draw.rectangle([cx - half, cy - half, cx + half, cy + half],
                           outline=(255, 0, 0, 120), width=1)
    # highlight PP rows (green) + result row (blue)
    for i in range(B):
        cy = pad + pp_row(i) * cs + cs / 2
        draw.line([(0, cy), (img.size[0], cy)], fill=(0, 200, 0, 160), width=1)
    ry = pad + result_row(B) * cs + cs / 2
    draw.line([(0, ry), (img.size[0], ry)], fill=(0, 0, 255, 200), width=2)
    img.save(out_path)
    return out_path


# ───────────────────────── self-check ────────────────────────────────────────


def _selfcheck() -> None:
    """Cross-validate pp_row/result_row + build_expected against create_layout's
    actual cells for A,B ∈ [1,6]. A generated target judged against itself must
    score partial_product_acc=1.0 and total_correct=1; erasing a GT digit cell
    must flip that PP (and the total) to wrong."""
    import random

    rng = random.Random(0)
    ok = True
    for A in range(1, 7):
        for B in range(1, 7):
            a = rng.randint(10 ** (A - 1), 10 ** A - 1)
            b = rng.randint(10 ** (B - 1), 10 ** B - 1)
            result = compute_multiplication(a, b, 10)
            layout = create_layout(result, ColumnMulConfig(num_digits_a=A, num_digits_b=B, base=10), input_only=False)

            def _actual(row: int) -> set[tuple[int, str]]:
                return {(c.col, c.char) for c in layout.cells if c.row == row and not c.is_carry}

            for i in range(B):
                shift = i
                pp = result.partial_products[i]
                all_zero = all(d.digit == 0 for d in pp.digits)
                if all_zero:
                    expected_set = {(shift, "0")}  # _add_digit_cells collapses all-zero → single "0"
                else:
                    expected_set = {(shift + d.col, digit_to_char(d.digit)) for d in pp.digits}
                actual_set = _actual(pp_row(i))
                if expected_set != actual_set:
                    print(f"  PP MISMATCH A={A} B={B} i={i} row={pp_row(i)} expected={sorted(expected_set)} actual={sorted(actual_set)}")
                    ok = False

            rdigits = result.additions[-1].digits if B >= 2 else result.partial_products[0].digits
            expected_result_set = {(d.col, digit_to_char(d.digit)) for d in rdigits}
            actual_result_set = _actual(result_row(B))
            if expected_result_set != actual_result_set:
                print(f"  RESULT MISMATCH A={A} B={B} row={result_row(B)} expected={sorted(expected_result_set)} actual={sorted(actual_result_set)}")
                ok = False
    print(f"row cross-validation: {'OK' if ok else 'FAIL'} (A,B in [1,6])")

    # self-consistency: render a GT target, judge it against itself → full score.
    from renderer import render_image  # noqa: WPS433 (path already on sys.path)
    a, b, base = 123, 45, 10
    cfg = ColumnMulConfig(num_digits_a=3, num_digits_b=2, base=base)
    target = render_image(create_layout(compute_multiplication(a, b, base), cfg, input_only=False), cfg)
    import tempfile
    tpred = os.path.join(tempfile.gettempdir(), "_pp_selfcheck_pred.png")
    target.save(tpred)  # pred == target (perfect generation)
    clf = TemplateClassifier(base=base, cs=cfg.cell_size, font_name=cfg.font_name)
    res = judge_sample(tpred, tpred, a, b, base, clf)
    print(f"self-judge (GT as pred): pp_acc={res['partial_product_acc']:.3f} total={res['total_correct']} "
          f"(expect 1.0 / 1)")

    # truncation: blank out P_1's HIGHEST ACTUAL GT-digit cell (not the blank
    # overflow slot) → that PP must flip to wrong, and so must the total.
    A, B = 3, 2
    cs, pad = cfg.cell_size, cfg.padding
    num_cols = A + B + 1
    exp = build_expected(a, b, base, cs, pad)
    p1 = exp.pp[1]  # P_1 span = [(col, gt_char|None), ...]
    # highest col whose GT is an actual digit (skip trailing None/blank overflow slot)
    hi_col = max(col for col, gt in p1 if gt is not None)
    cx, cy = cell_center(pp_row(1), hi_col, num_cols, cs, pad)
    arr = np.asarray(target.convert("L")).copy()
    half = cs
    x0, y0 = int(cx - half), int(cy - half)
    arr[max(0, y0):y0 + 2 * cs, max(0, x0):x0 + 2 * cs] = 255  # paint the cell white
    Image.fromarray(arr).save(tpred)
    res2 = judge_sample(tpred, tpred, a, b, base, clf)
    print(f"self-judge (P_1 high GT-digit erased): pp_correct={res2['pp_correct']} total={res2['total_correct']} "
          f"(expect P_1=0; total unchanged — only P_1's row was erased, the result row is separate)")
    print("self-check done.")


if __name__ == "__main__":
    _selfcheck()
