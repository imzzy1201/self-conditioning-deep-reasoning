import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

from config import ColumnMulConfig, DEFAULT_FONT_NAME, round_up_to_multiple
from layout import Layout, Cell, HLine

# ───────────────────────── font resolution ──────────────────────────────────
# The dataset is rendered with a bundled mono font (DejaVu Sans Mono, free
# license) rather than an OS-installed font, so images are byte-identical on
# every machine regardless of installed fonts. This is what the template digit
# judge (src/image_only/partial_product_judge.py) keys on: generator and judge must
# load the SAME .ttf.
_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
# Display name -> bundled .ttf file. Resolved FIRST, before any OS font lookup.
_BUNDLED_FONTS = {
    DEFAULT_FONT_NAME: "DejaVuSansMono.ttf",
}

# OS font lookups below are only a last-resort fallback for non-default names
# (e.g. an explicit --font_name Consolas); the default never reaches them.
_WINDOWS_FONTS = "C:/Windows/Fonts"
_WINDOWS_FONT_MAP = {
    "Consolas": "consola.ttf",
    "Courier New": "cour.ttf",
    "Lucida Console": "lucon.ttf",
    "DejaVu Sans Mono": "DejaVuSansMono.ttf",
    "Liberation Mono": "LiberationMono-Regular.ttf",
}
_LINUX_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
]


def _load_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    # 1) Repo-bundled font (machine-independent; generator and judge agree).
    if font_name in _BUNDLED_FONTS:
        path = os.path.join(_BUNDLED_DIR, _BUNDLED_FONTS[font_name])
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            pass

    # 2) Direct name/path (fontconfig on Linux, or an explicit .ttf path).
    try:
        return ImageFont.truetype(font_name, font_size)
    except OSError:
        pass

    # 3) Windows Fonts dir (by known name).
    if font_name in _WINDOWS_FONT_MAP:
        path = os.path.join(_WINDOWS_FONTS, _WINDOWS_FONT_MAP[font_name])
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            pass

    # 4) Windows Fonts dir (by flattened name, e.g. "consolas.ttf").
    fallback = os.path.join(_WINDOWS_FONTS, font_name.lower().replace(" ", "") + ".ttf")
    try:
        return ImageFont.truetype(fallback, font_size)
    except OSError:
        pass

    # 5) Standard Linux system-font locations.
    for path in _LINUX_FONT_PATHS:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            pass

    raise RuntimeError(
        f"Cannot load font '{font_name}'. Install a monospace font or specify a valid font path."
    )


def _cell_center(
    row: int,
    col: int,
    num_cols: int,
    cell_size: int,
    padding: int,
) -> Tuple[float, float]:
    x = padding + (num_cols - 1 - col) * cell_size + cell_size / 2
    y = padding + row * cell_size + cell_size / 2
    return x, y


def _line_pixel_coords(
    line: HLine,
    num_cols: int,
    cell_size: int,
    padding: int,
) -> Tuple[float, float, float]:
    x_start = padding + (num_cols - 1 - line.col_left) * cell_size
    x_end = padding + (num_cols - 1 - line.col_right) * cell_size + cell_size
    y = padding + line.row * cell_size + cell_size * 2 // 3
    return x_start, x_end, y


def render_image(layout: Layout, config: ColumnMulConfig) -> Image.Image:
    cell_size = config.cell_size
    padding = config.padding
    line_width = config.line_width if config.line_width is not None else max(cell_size // 16, 1)

    raw_width = padding * 2 + layout.num_cols * cell_size
    raw_height = padding * 2 + layout.num_rows * cell_size

    if config.image_width is not None:
        img_width = max(config.image_width, raw_width)
    else:
        img_width = raw_width
    if config.image_height is not None:
        img_height = max(config.image_height, raw_height)
    else:
        img_height = raw_height

    img_width = round_up_to_multiple(img_width, 16)
    img_height = round_up_to_multiple(img_height, 16)

    img = Image.new("RGB", (img_width, img_height), config.background_color)
    draw = ImageDraw.Draw(img)

    font_size_normal = int(0.65 * 1.25 * cell_size)
    font_size_carry = max(int(0.65 * 1.25 * cell_size / 2 * 1.5), 6)
    font_normal = _load_font(config.font_name, font_size_normal)
    font_carry = _load_font(config.font_name, font_size_carry)

    # Draw horizontal lines
    for line in layout.lines:
        x_start, x_end, y = _line_pixel_coords(
            line, layout.num_cols, cell_size, padding
        )
        draw.line([(x_start, y), (x_end, y)], fill=config.line_color, width=line_width)

    # Draw cells: normal digits first, then carries (so carries render on top)
    normal_cells = [c for c in layout.cells if not c.is_carry]
    carry_cells = [c for c in layout.cells if c.is_carry]

    for cell in normal_cells:
        cx, cy = _cell_center(cell.row, cell.col, layout.num_cols, cell_size, padding)
        draw.text(
            (cx, cy),
            cell.char,
            font=font_normal,
            fill=config.foreground_color,
            anchor="mm",
        )

    for cell in carry_cells:
        cx, cy = _cell_center(cell.row, cell.col, layout.num_cols, cell_size, padding)
        carry_cx = cx - cell_size / 2
        carry_cy = cy - cell_size / 4
        draw.text(
            (carry_cx, carry_cy),
            cell.char,
            font=font_carry,
            fill=config.foreground_color,
            anchor="mm",
        )

    return img