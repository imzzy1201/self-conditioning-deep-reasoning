import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# --- Cross-package import for text rendering ---
# Add src/ to sys.path so text_dataset can be imported as a package.
_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from text_dataset.rendering import load_font as _load_font, measure_text_height, render_text_image


def resolve_data_root(metadata_csv: str, data_root: str | None) -> Path:
    metadata_path = Path(metadata_csv)
    return Path(data_root) if data_root is not None else metadata_path.parent


def load_metadata_rows(metadata_csv: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(metadata_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "image" not in row or "edit_image" not in row:
                raise ValueError("metadata.csv must contain columns: image, edit_image, prompt")
            rows.append(row)
    if len(rows) == 0:
        raise ValueError("metadata.csv is empty")
    return rows


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def center_crop_to_divisor(img: Image.Image, divisor: int) -> Image.Image:
    if divisor <= 1:
        return img

    width, height = img.size
    width = width - (width % divisor)
    height = height - (height % divisor)
    if width <= 0 or height <= 0:
        raise ValueError("Image is too small after divisor adjustment")

    if (width, height) != img.size:
        left = (img.size[0] - width) // 2
        top = (img.size[1] - height) // 2
        img = img.crop((left, top, left + width, top + height))

    return img


def load_rgb_image(path: str | Path) -> Image.Image:
    """Load an RGB PIL Image from a file path."""
    with Image.open(path) as img:
        return img.convert("RGB")


# --- Shard internals ---
# Shard URL format: "shard:NNNNNN:idx"
# NNNNNN = shard start index, idx = local index within shard
# Both "image" and "edit_image" columns use the same URL;
# the Dataset reads from "outputs" or "inputs" tensor based on key.

SHARD_URL_PREFIX = "shard:"


def _parse_shard_url(url: str) -> tuple[int, int]:
    parts = url[len(SHARD_URL_PREFIX):].split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard URL format: {url}")
    return int(parts[0]), int(parts[1])


def _shard_url_to_filename(shard_start: int) -> str:
    return f"shard_{shard_start:06d}.pt"


def _resolve_path(path_str: str, data_root: Path) -> Path:
    """Resolve a relative file path against data_root. Only for PNG mode."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return data_root / path


class ShardCache:
    """Lazy-loading cache for shard .pt files.

    Keeps up to max_cached shards in memory. When a new shard is requested
    beyond the limit, the oldest entry is evicted (LRU).

    preload_count=N: loads the first N shards at init time so __getitem
    avoids torch.load for those shards. When N < total shards, data
    should be accessed sequentially (no shuffle) to avoid thrashing.
    preload_all=True: loads every shard (equivalent to preload_count=total).
    """

    def __init__(self, shard_dir: Path, max_cached: int = 2,
                 preload_all: bool = False, preload_count: int = 0):
        self.shard_dir = shard_dir
        self.max_cached = max_cached
        self._cache: dict[int, dict] = {}

        # Read shard index to know all available shard start indices
        index_path = shard_dir / "shard_index.json"
        with index_path.open("r") as f:
            self._shard_index = json.load(f)
        self._all_starts = sorted(
            int(name.split("_")[1].split(".")[0]) for name in self._shard_index["shards"]
        )

        if preload_all:
            for shard_name in self._shard_index["shards"]:
                start = int(shard_name.split("_")[1].split(".")[0])
                shard_path = shard_dir / shard_name
                self._cache[start] = torch.load(shard_path, weights_only=False)
            self.max_cached = len(self._cache)
        elif preload_count > 0:
            for start in self._all_starts[:preload_count]:
                shard_path = shard_dir / _shard_url_to_filename(start)
                self._cache[start] = torch.load(shard_path, weights_only=False)
            self.max_cached = max(self.max_cached, preload_count)

    @property
    def total_shards(self) -> int:
        return len(self._all_starts)

    def load(self, shard_start: int) -> dict:
        if shard_start not in self._cache:
            filename = _shard_url_to_filename(shard_start)
            shard_path = self.shard_dir / filename
            data = torch.load(shard_path, weights_only=False)
            self._cache[shard_start] = data
            if len(self._cache) > self.max_cached:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
        return self._cache[shard_start]

    def get_pil_image(self, url: str, key: str) -> Image.Image:
        """Load a PIL Image from a shard URL.

        key: "inputs" for condition image, "outputs" for target image.
        Supports both PNG-bytes format (compact) and uint8-tensor format (legacy).
        """
        shard_start, local_idx = _parse_shard_url(url)
        data = self.load(shard_start)
        entry = data[key][local_idx]

        if isinstance(entry, bytes):
            return Image.open(io.BytesIO(entry)).convert("RGB")
        else:
            arr = entry.permute(1, 2, 0).numpy().astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")


def _detect_shard_mode(data_root: Path, preload_all: bool = False,
                       preload_count: int = 0) -> ShardCache | None:
    """Check for shard files at data_root and return a ShardCache if found."""
    shard_dir = data_root / "shards"
    shard_index_path = shard_dir / "shard_index.json"
    if shard_dir.is_dir() and shard_index_path.exists():
        return ShardCache(shard_dir, preload_all=preload_all, preload_count=preload_count)
    return None


class MetadataImageEditDataset(Dataset):
    """Loads image-edit pairs from a metadata.csv.

    Supports two storage modes, auto-detected at init:
    - PNG mode: individual PNG files referenced by relative paths in metadata.csv
    - Shard mode: shard .pt files referenced by shard:NNNNNN:idx URLs in metadata.csv

    Two access patterns:
    - __getitem__(idx): returns [-1, 1] float tensors for training
    - get_pil_sample(idx): returns RGB PIL Images + metadata dict for evaluation
    """

    def __init__(self, metadata_csv: str, data_root: str | None, size_divisor: int = 16,
                 preload_shards: bool = False, preload_shards_count: int = 0):
        self.metadata_csv = Path(metadata_csv)
        self.data_root = resolve_data_root(metadata_csv, data_root)
        self.size_divisor = size_divisor
        self.rows = load_metadata_rows(metadata_csv)

        # Auto-detect shard mode
        self._shard_cache = _detect_shard_mode(
            self.data_root, preload_all=preload_shards, preload_count=preload_shards_count
        )
        self._shard_mode = self._shard_cache is not None

    def __len__(self):
        return len(self.rows)

    # --- PIL Image access (for evaluation) ---

    def get_pil_sample(self, idx: int) -> dict[str, Any]:
        """Return a sample as RGB PIL Images + metadata dict.

        Returns {"image": PIL Image, "edit_image": PIL Image, "metadata": dict}.
        """
        row = self.rows[idx]

        if self._shard_mode:
            image_url = row["image"]
            edit_image_url = row["edit_image"]
            image = self._shard_cache.get_pil_image(image_url, "outputs")
            edit_image = self._shard_cache.get_pil_image(edit_image_url, "inputs")
        else:
            image_path = _resolve_path(row["image"], self.data_root)
            edit_image_path = _resolve_path(row["edit_image"], self.data_root)

            if not image_path.exists():
                raise FileNotFoundError(f"Missing target image: {image_path}")
            if not edit_image_path.exists():
                raise FileNotFoundError(f"Missing condition image: {edit_image_path}")

            image = load_rgb_image(image_path)
            edit_image = load_rgb_image(edit_image_path)

        return {
            "image": image,
            "edit_image": edit_image,
            "metadata": row,
        }

    # --- Tensor access (for training) ---

    def _load_from_shard(self, url: str, key: str) -> torch.Tensor:
        """Load an image tensor from a shard URL.

        key: "inputs" → condition image, "outputs" → target image.
        Returns [-1, 1] float tensor [3, H, W].
        Supports both PNG-bytes format (compact) and uint8-tensor format (legacy).
        """
        shard_start, local_idx = _parse_shard_url(url)
        data = self._shard_cache.load(shard_start)
        entry = data[key][local_idx]

        if isinstance(entry, bytes):
            img = Image.open(io.BytesIO(entry)).convert("RGB")
            return pil_to_tensor(img)
        else:
            return entry.float() / 127.5 - 1.0

    def _load_target_image(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = center_crop_to_divisor(img, self.size_divisor)
            target_size = img.size
            tensor = pil_to_tensor(img)
        return tensor, target_size

    def _load_condition_image(self, path: Path, target_size: tuple[int, int]) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size != target_size:
                img = ImageOps.fit(img, target_size, method=Image.Resampling.BICUBIC)
            return pil_to_tensor(img)

    def __getitem__(self, idx):
        row = self.rows[idx]

        if self._shard_mode:
            image_url = row["image"]
            edit_image_url = row["edit_image"]
            image = self._load_from_shard(image_url, "outputs")
            edit_image = self._load_from_shard(edit_image_url, "inputs")
        else:
            image_path = _resolve_path(row["image"], self.data_root)
            edit_image_path = _resolve_path(row["edit_image"], self.data_root)

            if not image_path.exists():
                raise FileNotFoundError(f"Missing target image: {image_path}")
            if not edit_image_path.exists():
                raise FileNotFoundError(f"Missing condition image: {edit_image_path}")

            image, target_size = self._load_target_image(image_path)
            edit_image = self._load_condition_image(edit_image_path, target_size)

        return {
            "image": image,
            "edit_image": edit_image,
        }


def collate_image_edit_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    shapes = {tuple(item["image"].shape) for item in batch}
    if len(shapes) != 1:
        raise ValueError(
            "Variable image sizes detected in one batch. Use batch_size=1 or ensure all images have the same resolution."
        )

    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "edit_image": torch.stack([item["edit_image"] for item in batch], dim=0),
    }


# --- On-the-fly text-to-image dataset ---


def _tinystories_extract(text: str) -> dict[str, str]:
    """Extract prompt (first paragraph) and response (full text) from TinyStories."""
    text = str(text) if text is not None else ""
    paragraphs = text.strip().split("\n\n")
    prompt = paragraphs[0].strip() if paragraphs else text.strip()
    response = text.strip()
    return {"prompt": prompt, "response": response}


class TextImageEditDataset(Dataset):
    """On-the-fly text-to-image dataset for flow matching training.

    Loads a HuggingFace text dataset and renders each sample's prompt/response
    text into images at __getitem__ time, avoiding the need for pre-rendering.
    Returns the same format as MetadataImageEditDataset:
    - __getitem__(idx): {"image": tensor[-1,1], "edit_image": tensor[-1,1]}
    - get_pil_sample(idx): {"image": PIL, "edit_image": PIL, "metadata": dict}
    """

    def __init__(
        self,
        dataset_name: str | None = None,
        split: str = "train",
        dataset: Any | None = None,
        text_column: str = "text",
        prompt_column: str | None = None,
        response_column: str | None = None,
        extract_fn: Callable[[str], dict[str, str]] | None = None,
        image_width: int = 512,
        image_height: int = 512,
        padding: int = 16,
        font_name: str = "Consolas",
        font_size: int = 13,
        line_spacing: int = 4,
        paragraph_gap_lines: int = 1,
        size_divisor: int = 16,
        hf_endpoint: str | None = None,
        max_samples: int | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if image_width % 16 != 0:
            raise ValueError("image_width must be a multiple of 16")
        if image_height % 16 != 0:
            raise ValueError("image_height must be a multiple of 16")

        # --- Load dataset ---
        if dataset_name is not None and dataset is not None:
            raise ValueError("Provide either dataset_name or dataset, not both")
        if dataset_name is None and dataset is None:
            raise ValueError("Either dataset_name or dataset must be provided")

        if dataset is not None:
            hf_dataset = dataset
        else:
            if hf_endpoint is not None:
                os.environ["HF_ENDPOINT"] = hf_endpoint
            from datasets import load_dataset
            hf_dataset = load_dataset(dataset_name, split=split)

        if shuffle:
            hf_dataset = hf_dataset.shuffle(seed=seed)
        if max_samples is not None:
            num = min(max_samples, len(hf_dataset))
            hf_dataset = hf_dataset.select(range(num))

        # --- Extract prompt/response text ---
        if (prompt_column is not None) != (response_column is not None):
            raise ValueError("prompt_column and response_column must both be provided or both be None")
        if prompt_column is not None and response_column is not None:
            # Dataset already has prompt/response columns — read them directly.
            pcol = prompt_column
            rcol = response_column
        else:
            fn = extract_fn or _tinystories_extract
            hf_dataset = hf_dataset.map(lambda row: fn(row[text_column]))
            pcol = "prompt"
            rcol = "response"

        self._samples: list[tuple[str, str]] = [
            (str(row[pcol]), str(row[rcol])) for row in hf_dataset
        ]

        # --- Rendering parameters (stored as plain values for pickle; font loaded lazily) ---
        self.image_width = image_width
        self.image_height = image_height
        self.padding = padding
        self.font_name = font_name
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.paragraph_gap_lines = paragraph_gap_lines
        self.size_divisor = size_divisor

        self._font: Any = None  # lazy-loaded in _ensure_font
        self._line_height: int | None = None

    def _ensure_font(self):
        """Lazy-load font object. Called on first __getitem__ in each process."""
        if self._font is None:
            self._font = _load_font(self.font_name, self.font_size)
            self._line_height = measure_text_height(self._font) + self.line_spacing

    def __len__(self):
        return len(self._samples)

    def _render_sample(self, idx: int) -> tuple[Image.Image, Image.Image]:
        """Render prompt and response text into PIL Images."""
        self._ensure_font()
        prompt_text, response_text = self._samples[idx]
        edit_img = render_text_image(
            prompt_text, self.image_width, self.image_height,
            self._font, self._line_height, self.padding, self.paragraph_gap_lines,
        )
        target_img = render_text_image(
            response_text, self.image_width, self.image_height,
            self._font, self._line_height, self.padding, self.paragraph_gap_lines,
        )
        return edit_img, target_img

    def get_pil_sample(self, idx: int) -> dict[str, Any]:
        """Return a sample as RGB PIL Images + metadata dict."""
        edit_img, target_img = self._render_sample(idx)
        prompt_text, response_text = self._samples[idx]
        return {
            "image": target_img,
            "edit_image": edit_img,
            "metadata": {
                "prompt": prompt_text,
                "response": response_text,
                "index": idx,
                "image": f"text_{idx:06d}",
                "edit_image": f"text_{idx:06d}",
            },
        }

    def __getitem__(self, idx):
        edit_img, target_img = self._render_sample(idx)

        target_img = center_crop_to_divisor(target_img, self.size_divisor)
        target_size = target_img.size
        if edit_img.size != target_size:
            edit_img = ImageOps.fit(edit_img, target_size, method=Image.Resampling.BICUBIC)

        return {
            "image": pil_to_tensor(target_img),
            "edit_image": pil_to_tensor(edit_img),
        }
