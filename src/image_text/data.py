import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


# Make the Edit2Perceive perception-normalization cores (in the sibling
# `depth_normal_estimation` package) importable. Imported lazily only when
# target_mode="perception" is used, so plain RGB training needs neither this
# path nor cv2.
_PERCEPTION_CONV_DIR = str(Path(__file__).resolve().parents[1] / "depth_normal_estimation")
if _PERCEPTION_CONV_DIR not in sys.path:
    sys.path.insert(0, _PERCEPTION_CONV_DIR)

# source_dataset -> (divisor, lo, hi) for raw depth normalization (E2P vanilla,
# no log/sqrt/pdf). Matches external/Edit2Perceive/models/unified_dataset.py:466-471.
_DEPTH_NORM = {
    "hypersim": (1000.0, 1e-3, 65.5),   # uint16 mm -> m; valid (1mm, 65.5m)
    "vkitti":   (100.0,  0.1,   80.0),  # uint16 cm -> m; valid (0.1m, 80m)
}


def resolve_data_root(metadata_csv: str, data_root: str | None) -> Path:
    metadata_path = Path(metadata_csv)
    return Path(data_root) if data_root is not None else metadata_path.parent


def load_metadata_rows(metadata_csv: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(metadata_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "image" not in row:
                raise ValueError("metadata.csv must contain at least column: image")
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
                 preload_shards: bool = False, preload_shards_count: int = 0,
                 target_mode: str = "rgb", target_height: int | None = None,
                 target_width: int | None = None, depth_normalization: str = "log_percentile",
                 max_size: int | None = None):
        self.metadata_csv = Path(metadata_csv)
        self.data_root = resolve_data_root(metadata_csv, data_root)
        self.size_divisor = size_divisor
        # RGB mode: downscale any image whose longest side exceeds max_size to
        # <= max_size (preserve aspect, bicubic; only downscales). None = off.
        # Applied to target + condition in both tensor and PIL (eval) paths.
        self.max_size = max_size
        self.rows = load_metadata_rows(metadata_csv)

        # Perception mode: load raw numeric GT (uint16 depth PNG / float32 normal
        # .npy) and normalize on-the-fly -- FE2E eq 7 for depth (log + per-image
        # 2-98 percentile + BF16 truncation), minimal for normal. See
        # `_load_target_perception`. RGB mode (default) is unchanged.
        self.target_mode = target_mode
        self.target_height = target_height
        self.target_width = target_width
        self.depth_normalization = depth_normalization
        if target_mode == "perception":
            if depth_normalization not in ("log_percentile", "percentile", "none"):
                raise ValueError(f"depth_normalization must be 'log_percentile'|'percentile'|'none', got {depth_normalization!r}")
            # target_height/width may be None -> dynamic center_crop_to_divisor
            # (like RGB mode; requires batch_size=1 or uniform-source batches).
            # Set them (e.g. 768x768) for fixed-size mixed batches.

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

        Returns {"image": PIL Image, "edit_image": PIL Image or None, "metadata": dict}.
        edit_image is None when the CSV row has an empty edit_image value.
        """
        if self.target_mode == "perception":
            raise NotImplementedError(
                "get_pil_sample is not implemented for target_mode='perception' "
                "(raw depth/normal targets). Perception-target eval is a follow-up; "
                "this dataset is training-only for now."
            )
        row = self.rows[idx]
        edit_image_val = row.get("edit_image", "").strip()

        if self._shard_mode:
            image_url = row["image"]
            image = self._shard_cache.get_pil_image(image_url, "outputs")
            edit_image = None
            if edit_image_val:
                edit_image = self._shard_cache.get_pil_image(edit_image_val, "inputs")
        else:
            image_path = _resolve_path(row["image"], self.data_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing target image: {image_path}")
            image = load_rgb_image(image_path)
            edit_image = None
            if edit_image_val:
                edit_image_path = _resolve_path(edit_image_val, self.data_root)
                if not edit_image_path.is_file():
                    raise FileNotFoundError(f"Missing condition image: {edit_image_path}")
                edit_image = load_rgb_image(edit_image_path)

        image = self._apply_max_size(image)
        if edit_image is not None:
            edit_image = self._apply_max_size(edit_image)
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

    def _apply_max_size(self, img: Image.Image) -> Image.Image:
        """Downscale so the longest side <= self.max_size (preserve aspect, bicubic).

        No-op when max_size is None or the image is already <= max_size on both
        sides. Mutates in place (the caller passes a freshly-loaded image) and
        returns it. Used so high-res variable-resolution datasets fit GPU memory.
        """
        if self.max_size and max(img.size) > self.max_size:
            img.thumbnail((self.max_size, self.max_size), Image.Resampling.BICUBIC)
        return img

    def _load_target_image(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self._apply_max_size(img)
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

    # --- Perception mode (raw depth/normal GT, on-the-fly normalize) ---

    def _size_pil(self, img: Image.Image, resample=None) -> Image.Image:
        """Fixed resize to target_height/width if set, else center_crop_to_divisor.

        `resample=None` -> bare `img.resize((W,H))` (PIL default; matches E2P's
        16-bit depth resize). Pass BILINEAR for RGB conditions.
        """
        if self.target_height and self.target_width:
            if resample is None:
                return img.resize((self.target_width, self.target_height))
            return img.resize((self.target_width, self.target_height), resample)
        return center_crop_to_divisor(img, self.size_divisor)

    def _size_array(self, arr: np.ndarray) -> np.ndarray:
        """Fixed cv2 resize (INTER_LINEAR_EXACT) if target size set, else center crop."""
        if self.target_height and self.target_width:
            import cv2  # noqa: E402  (only needed for normal targets)
            return cv2.resize(arr, (self.target_width, self.target_height),
                              interpolation=cv2.INTER_LINEAR_EXACT)
        h, w = arr.shape[:2]
        nh = h - (h % self.size_divisor)
        nw = w - (w % self.size_divisor)
        if nh <= 0 or nw <= 0:
            raise ValueError("normal array too small after divisor adjustment")
        top = (h - nh) // 2
        left = (w - nw) // 2
        return arr[top:top + nh, left:left + nw]

    def _load_target_perception(self, row: dict[str, Any]) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load a raw depth/normal GT and normalize on-the-fly -> [3,H,W] in [-1,1].

        Depth: FE2E eq 7 -- D_log=ln(D+1e-6), per-image 2-98 percentile on D_log,
        stretch to [-1,1], BF16-truncate, 3ch replicate. (method='percentile'
        falls back to legacy linear percentile; 'none' to a linear lo/hi map.)
        Normal: minimal (load + resize + clip) -> already [-1,1] 3ch; no x-flip/renorm.
        """
        from conversions import normalize_depth, normalize_normal_minimal  # noqa: E402

        task = row.get("task", "").strip()
        image_path = _resolve_path(row["image"], self.data_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing perception target: {image_path}")

        if task == "depth":
            src = row.get("source_dataset", "hypersim").strip()
            divisor, lo, hi = _DEPTH_NORM.get(src, _DEPTH_NORM["hypersim"])
            img = Image.open(image_path)  # uint16 "I;16"; bare resize matches E2P
            img = self._size_pil(img)
            arr = np.array(img, dtype=np.float32)
            arr3 = normalize_depth(arr, divisor, lo, hi, method=self.depth_normalization)
        elif task == "normal":
            arr = np.load(image_path).astype(np.float32)  # (H,W,3) unit vectors in [-1,1]
            arr = self._size_array(arr)
            arr3 = normalize_normal_minimal(arr)
        else:
            raise ValueError(f"Unknown task {task!r} in perception mode (row: {row})")

        tensor = torch.from_numpy(np.ascontiguousarray(arr3)).permute(2, 0, 1).contiguous()
        target_size = (tensor.shape[2], tensor.shape[1])  # (W, H)
        return tensor, target_size

    def _load_condition_perception(self, path: Path) -> torch.Tensor:
        """RGB condition resized/center-cropped with the same rule as the target."""
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self._size_pil(img, Image.Resampling.BILINEAR)
            return pil_to_tensor(img)

    def __getitem__(self, idx):
        row = self.rows[idx]
        edit_image_val = row.get("edit_image", "").strip()

        if self.target_mode == "perception":
            image, _ = self._load_target_perception(row)
            edit_image = None
            if edit_image_val:
                edit_image_path = _resolve_path(edit_image_val, self.data_root)
                if not edit_image_path.is_file():
                    raise FileNotFoundError(f"Missing condition image: {edit_image_path}")
                edit_image = self._load_condition_perception(edit_image_path)
            return {"image": image, "edit_image": edit_image, "prompt": row.get("prompt", "")}

        if self._shard_mode:
            image_url = row["image"]
            image = self._load_from_shard(image_url, "outputs")
            edit_image = None
            if edit_image_val:
                edit_image = self._load_from_shard(edit_image_val, "inputs")
        else:
            image_path = _resolve_path(row["image"], self.data_root)
            if not image_path.exists():
                raise FileNotFoundError(f"Missing target image: {image_path}")

            image, target_size = self._load_target_image(image_path)
            edit_image = None
            if edit_image_val:
                edit_image_path = _resolve_path(edit_image_val, self.data_root)
                if not edit_image_path.is_file():
                    raise FileNotFoundError(f"Missing condition image: {edit_image_path}")
                edit_image = self._load_condition_image(edit_image_path, target_size)

        return {
            "image": image,
            "edit_image": edit_image,
            "prompt": row.get("prompt", ""),
        }


def collate_image_edit_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor | list[str] | None]:
    shapes = {tuple(item["image"].shape) for item in batch}
    if len(shapes) != 1:
        raise ValueError(
            "Variable image sizes detected in one batch. Use batch_size=1 or ensure all images have the same resolution."
        )

    # edit_image: all present or all absent
    has_edit = [item.get("edit_image") is not None for item in batch]
    if any(has_edit) and not all(has_edit):
        raise ValueError(
            "Mixed edit_image in batch: some None, some present. "
            "All items must have edit_image or none must."
        )

    result: dict = {
        "image": torch.stack([item["image"] for item in batch], dim=0),
    }
    if all(has_edit):
        result["edit_image"] = torch.stack([item["edit_image"] for item in batch], dim=0)
    else:
        result["edit_image"] = None
    result["prompts"] = [item.get("prompt", "") for item in batch]
    return result
