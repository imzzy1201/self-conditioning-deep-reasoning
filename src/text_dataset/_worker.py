import io

from PIL import Image

try:
    from .rendering import load_font, measure_text_height, render_text_image
except ImportError:
    from rendering import load_font, measure_text_height, render_text_image


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_sample(args: tuple):
    """Render a single sample's images and return PNG bytes (worker entry point).

    Returns PNG bytes instead of PIL Images to avoid ~7MB/sample memory
    overhead from executor.map() holding unpickled PIL Images in completed
    Futures. PNG bytes are ~30KB vs PIL Image ~0.75MB each.
    """
    (idx, prompt_text, prompt_prefix, font_name, font_size,
     image_width, image_height, padding, line_spacing, paragraph_gap_lines) = args

    font = load_font(font_name, font_size)
    text_height = measure_text_height(font)
    line_height = text_height + line_spacing

    render_text = prompt_text[len(prompt_prefix):] if prompt_prefix and prompt_text.startswith(prompt_prefix) else prompt_text

    input_image = render_text_image(
        prompt_text, image_width, image_height, font, line_height, padding, paragraph_gap_lines
    )
    output_image = render_text_image(
        render_text, image_width, image_height, font, line_height, padding, paragraph_gap_lines
    )

    input_png = _pil_to_png_bytes(input_image)
    output_png = _pil_to_png_bytes(output_image)

    return idx, input_png, output_png, (image_width, image_height)


def render_sample_no_edit(args: tuple):
    """Render only the output image (no condition image). Returns empty input bytes.

    Used when --no_edit_image is set: prompt text is stored directly in metadata.csv
    instead of being rendered as a condition image.
    """
    (idx, prompt_text, prompt_prefix, font_name, font_size,
     image_width, image_height, padding, line_spacing, paragraph_gap_lines) = args

    font = load_font(font_name, font_size)
    text_height = measure_text_height(font)
    line_height = text_height + line_spacing

    render_text = prompt_text[len(prompt_prefix):] if prompt_prefix and prompt_text.startswith(prompt_prefix) else prompt_text

    output_image = render_text_image(
        render_text, image_width, image_height, font, line_height, padding, paragraph_gap_lines
    )
    output_png = _pil_to_png_bytes(output_image)

    return idx, b"", output_png, (image_width, image_height)