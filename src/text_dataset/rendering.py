import os

from PIL import Image, ImageDraw, ImageFont

try:
    from .utils import clean_text
except ImportError:
    from utils import clean_text


def load_font(font_name: str, font_size: int) -> ImageFont.ImageFont:
    if os.path.isfile(font_name):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            pass

    candidates = [
        font_name,
        "Consolas",
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
    ]

    windows_fonts = os.path.join("C:/Windows", "Fonts")
    windows_candidates = [
        os.path.join(windows_fonts, "consola.ttf"),
        os.path.join(windows_fonts, "cour.ttf"),
        os.path.join(windows_fonts, "lucon.ttf"),
    ]

    linux_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]

    for candidate in candidates + windows_candidates + linux_candidates:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def measure_text_width(font: ImageFont.ImageFont, text: str) -> int:
    content = text if text else " "
    return int(font.getlength(content))


def measure_text_height(font: ImageFont.ImageFont) -> int:
    left, top, right, bottom = font.getbbox("Ag")
    return bottom - top


def split_long_token(
    token: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""
    for ch in token:
        candidate = current + ch
        width = measure_text_width(font, candidate)
        if width <= max_width or not current:
            current = candidate
            continue
        chunks.append(current)
        current = ch
    if current:
        chunks.append(current)
    return chunks


def wrap_text(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []

    words = normalized.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        candidate_width = measure_text_width(font, candidate)
        if candidate_width <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        word_width = measure_text_width(font, word)
        if word_width <= max_width:
            current = word
            continue

        lines.extend(split_long_token(word, font, max_width))

    if current:
        lines.append(current)

    return lines


def wrap_text_with_paragraphs(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    paragraph_gap_lines: int = 1,
) -> list[str]:
    """Wrap text preserving paragraph breaks as empty lines.

    Paragraphs are separated by \\n\\n. Each paragraph is wrapped
    independently. Between paragraphs, `paragraph_gap_lines` empty
    strings are inserted to create visual spacing.
    """
    paragraphs = text.strip().split("\n\n")
    all_lines: list[str] = []
    for i, para in enumerate(paragraphs):
        para_text = clean_text(para)
        if not para_text:
            continue
        para_lines = wrap_text(para_text, font, max_width)
        if i > 0 and all_lines:
            all_lines.extend([""] * paragraph_gap_lines)
        all_lines.extend(para_lines)
    return all_lines


def render_text_image(
    text: str,
    width: int,
    height: int,
    font: ImageFont.ImageFont,
    line_height: int,
    padding: int = 16,
    paragraph_gap_lines: int = 1,
) -> Image.Image:
    """Render text onto a white image with auto-wrapping and truncation.

    Text that exceeds the available vertical space is truncated
    (lines beyond height - padding are simply not drawn).
    """
    lines = wrap_text_with_paragraphs(
        text, font, width - padding * 2, paragraph_gap_lines
    )

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    y = padding
    text_color = (0, 0, 0)
    max_y = height - padding

    for line in lines:
        if y + line_height > max_y:
            break
        if line:
            draw.text((padding, y), line, font=font, fill=text_color)
        y += line_height

    return image