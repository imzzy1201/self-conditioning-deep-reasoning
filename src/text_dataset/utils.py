import os
from typing import Any


def round_up_to_multiple(value: int, multiple: int = 16) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def parse_image_height_arg(raw: str) -> int | None:
    value = raw.strip().lower()
    if value == "pad":
        return None
    try:
        iv = int(raw)
    except ValueError as exc:
        raise SystemExit("--image_height must be an integer (multiple of 16) or 'pad'") from exc
    if iv <= 0:
        raise SystemExit("--image_height must be positive")
    if iv % 16 != 0:
        raise SystemExit("--image_height must be a multiple of 16")
    return iv


def parse_splits(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split name")
    return splits


def clean_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def rel_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")