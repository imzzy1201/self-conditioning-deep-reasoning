"""Render A5 permutation sequences as black or colored line traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from PIL import Image, ImageDraw

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from a5_line_trace_image_dataset.a5 import Permutation, prefix_compositions


Palette = dict[int, tuple[int, int, int]]

DEFAULT_PALETTE: Palette = {
    1: (230, 25, 75),
    2: (60, 180, 75),
    3: (0, 130, 200),
    4: (245, 130, 48),
    5: (145, 30, 180),
}

BACKGROUND_COLOR = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class Segment:
    start_row: int
    step: int
    left_row: int
    right_row: int
    color: tuple[int, int, int]


def active_columns_for_steps(num_permutations: int, column_stride: int = 4) -> int:
    if num_permutations <= 0:
        raise ValueError("num_permutations must be positive")
    if column_stride <= 0:
        raise ValueError("column_stride must be positive")
    return column_stride * num_permutations + 1


def point_for_cell(row: int, col: int, patch_size: int) -> tuple[int, int]:
    """Return pixel center for a 1-indexed row/column cell."""

    return (
        int(round((col - 0.5) * patch_size)),
        int(round((row - 0.5) * patch_size)),
    )


def trace_paths(sequence: list[Permutation]) -> list[list[int]]:
    """Return row paths, one path per first-column starting row."""

    paths: list[list[int]] = []
    for start_row in range(1, 6):
        row = start_row
        path = [row]
        for perm in sequence:
            row = int(perm[row - 1])
            path.append(row)
        paths.append(path)
    return paths


def trace_segments(sequence: list[Permutation], palette: Palette | None = None) -> list[Segment]:
    palette = palette or DEFAULT_PALETTE
    paths = trace_paths(sequence)
    segments: list[Segment] = []
    for start_row, path in enumerate(paths, start=1):
        color = palette[start_row]
        for step in range(1, len(path)):
            segments.append(
                Segment(
                    start_row=start_row,
                    step=step,
                    left_row=path[step - 1],
                    right_row=path[step],
                    color=color,
                )
            )
    return segments


def _validate_canvas(
    sequence: list[Permutation],
    patch_size: int,
    image_width: int,
    column_stride: int,
    line_width: int,
    horizontal_line_width: int,
    displacement3_line_width: int,
    node_diameter: int,
) -> None:
    if not sequence:
        raise ValueError("sequence must contain at least one permutation")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_width <= 0 or image_width % patch_size != 0:
        raise ValueError("image_width must be a positive multiple of patch_size")
    if column_stride <= 0:
        raise ValueError("column_stride must be positive")
    if image_width % 16 != 0 or (5 * patch_size) % 16 != 0:
        raise ValueError("image width and height must be multiples of 16")
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if horizontal_line_width <= 0:
        raise ValueError("horizontal_line_width must be positive")
    if displacement3_line_width <= 0:
        raise ValueError("displacement3_line_width must be positive")
    if node_diameter <= 0:
        raise ValueError("node_diameter must be positive")
    if active_columns_for_steps(len(sequence), column_stride) > image_width // patch_size:
        raise ValueError("sequence does not fit in the requested image width")


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    segment: Segment,
    patch_size: int,
    column_stride: int,
    line_width: int,
    horizontal_line_width: int,
    displacement3_line_width: int,
    color: tuple[int, int, int],
) -> None:
    left_col = 1 + column_stride * (segment.step - 1)
    right_col = 1 + column_stride * segment.step
    p0 = point_for_cell(segment.left_row, left_col, patch_size)
    p1 = point_for_cell(segment.right_row, right_col, patch_size)
    row_delta = abs(segment.right_row - segment.left_row)
    if row_delta == 0:
        width = horizontal_line_width
    elif row_delta == 3:
        width = displacement3_line_width
    else:
        width = line_width
    draw.line([p0, p1], fill=color, width=width)


def _draw_dot(
    draw: ImageDraw.ImageDraw,
    row: int,
    col: int,
    patch_size: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    x, y = point_for_cell(row, col, patch_size)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_condition_trace(
    sequence: list[Permutation],
    patch_size: int = 16,
    image_width: int = 1024,
    column_stride: int = 4,
    line_width: int = 4,
    horizontal_line_width: int = 3,
    displacement3_line_width: int = 3,
    node_diameter: int = 9,
    background_color: tuple[int, int, int] = BACKGROUND_COLOR,
) -> Image.Image:
    """Render raw permutation wiring as black dots and black line segments."""

    _validate_canvas(
        sequence,
        patch_size,
        image_width,
        column_stride,
        line_width,
        horizontal_line_width,
        displacement3_line_width,
        node_diameter,
    )
    image_height = 5 * patch_size
    img = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(img)
    dot_radius = node_diameter // 2

    for segment in trace_segments(sequence, {idx: BLACK for idx in range(1, 6)}):
        _draw_segment(
            draw,
            segment,
            patch_size,
            column_stride,
            line_width,
            horizontal_line_width,
            displacement3_line_width,
            BLACK,
        )

    for col in range(1, active_columns_for_steps(len(sequence), column_stride) + 1, column_stride):
        for row in range(1, 6):
            _draw_dot(draw, row, col, patch_size, dot_radius, BLACK)

    return img


def render_colored_trace(
    sequence: list[Permutation],
    patch_size: int = 16,
    image_width: int = 1024,
    column_stride: int = 4,
    line_width: int = 4,
    horizontal_line_width: int = 3,
    displacement3_line_width: int = 3,
    node_diameter: int = 9,
    palette: Palette | None = None,
    background_color: tuple[int, int, int] = BACKGROUND_COLOR,
) -> Image.Image:
    """Render paths colored by their first-column starting row."""

    _validate_canvas(
        sequence,
        patch_size,
        image_width,
        column_stride,
        line_width,
        horizontal_line_width,
        displacement3_line_width,
        node_diameter,
    )
    palette = palette or DEFAULT_PALETTE
    image_height = 5 * patch_size
    img = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(img)
    dot_radius = node_diameter // 2

    segments = trace_segments(sequence, palette)

    # Later strokes cover earlier strokes. Larger right_row-left_row means a
    # stronger downward displacement and is drawn above smaller displacements.
    segments = sorted(segments, key=lambda item: item.right_row - item.left_row)
    for segment in segments:
        _draw_segment(
            draw,
            segment,
            patch_size,
            column_stride,
            line_width,
            horizontal_line_width,
            displacement3_line_width,
            segment.color,
        )

    paths = trace_paths(sequence)
    for start_row, path in enumerate(paths, start=1):
        color = palette[start_row]
        for step, row in enumerate(path):
            col = 1 + column_stride * step
            _draw_dot(draw, row, col, patch_size, dot_radius, color)

    return img


def prefix_json_payload(sequence: list[Permutation]) -> list[list[int]]:
    return [list(perm) for perm in prefix_compositions(sequence)]
