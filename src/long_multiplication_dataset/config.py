from dataclasses import dataclass, field
from typing import Optional, Tuple

# Single default font shared by every entry point (see renderer._load_font).
# DejaVu Sans Mono ships with the repo at fonts/, so rendering is identical on
# every machine without relying on OS-installed fonts (generator == judge).
DEFAULT_FONT_NAME = "DejaVu Sans Mono"


def round_up_to_multiple(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def compute_max_digits_size(max_digits_a: int, max_digits_b: int, cell_size: int, padding: int) -> tuple[int, int]:
    """Compute image dimensions for max digit lengths (used by 'pad' mode).

    Returns (width, height) rounded up to multiples of 16.
    """
    from layout import _compute_total_rows
    num_cols = max_digits_a + max_digits_b + 1
    num_rows = _compute_total_rows(max_digits_b)
    raw_w = padding * 2 + num_cols * cell_size
    raw_h = padding * 2 + num_rows * cell_size
    return round_up_to_multiple(raw_w, 16), round_up_to_multiple(raw_h, 16)


def parse_image_size_arg(value: str | None, max_size: int | None, arg_name: str) -> int | None:
    """Parse --image_width/--image_height arg: int, 'pad', or None."""
    if value is None:
        return None
    if value.lower() == "pad":
        if max_size is None:
            raise SystemExit(f"--{arg_name} 'pad' failed to compute target size")
        return max_size
    try:
        iv = int(value)
    except ValueError:
        raise SystemExit(f"--{arg_name} must be an integer or 'pad', got: {value}")
    if iv % 16 != 0:
        raise SystemExit(f"--{arg_name} must be a multiple of 16")
    return iv


@dataclass
class ColumnMulConfig:
    num_digits_a: int = 3
    num_digits_b: int = 3
    base: int = 10
    cell_size: int = 16
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    font_name: str = DEFAULT_FONT_NAME
    padding: int = 16
    line_width: Optional[int] = None
    background_color: Tuple[int, int, int] = (255, 255, 255)
    foreground_color: Tuple[int, int, int] = (0, 0, 0)
    line_color: Tuple[int, int, int] = (0, 0, 0)
    num_samples: int = 1000
    output_dir: str = "output"
    seed: Optional[int] = None
