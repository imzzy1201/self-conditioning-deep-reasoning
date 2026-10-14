"""Shard writer compatible with ``MetadataImageEditDataset``."""

from __future__ import annotations

import io
import json
import os
from typing import Any

from PIL import Image


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ShardWriter:
    """Accumulate PNG bytes and write ``shard_*.pt`` files."""

    def __init__(self, output_dir: str, shard_size: int = 10000):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.shard_dir = os.path.join(output_dir, "shards")
        self.shard_size = shard_size
        os.makedirs(self.shard_dir, exist_ok=True)

        self._input_buf: list[bytes] = []
        self._output_buf: list[bytes] = []
        self._meta_buf: list[dict[str, Any]] = []
        self._global_idx = 0
        self._shard_names: list[str] = []
        self._image_width: int | None = None
        self._image_height: int | None = None

    def add(self, condition_image: Image.Image, target_image: Image.Image, metadata: dict[str, Any]) -> str:
        if self._image_width is None:
            self._image_width, self._image_height = target_image.size
        condition_bytes = pil_to_png_bytes(condition_image)
        target_bytes = pil_to_png_bytes(target_image)

        self._input_buf.append(condition_bytes)
        self._output_buf.append(target_bytes)
        self._meta_buf.append(metadata)

        shard_start = (self._global_idx // self.shard_size) * self.shard_size
        local_idx = self._global_idx % self.shard_size
        shard_name = f"shard_{shard_start:06d}.pt"
        shard_url = f"shard:{shard_start:06d}:{local_idx}"
        self._global_idx += 1

        if len(self._output_buf) == self.shard_size:
            self._flush(shard_name)
            self._shard_names.append(shard_name)

        return shard_url

    def finalize(self) -> None:
        if self._output_buf:
            shard_start = len(self._shard_names) * self.shard_size
            shard_name = f"shard_{shard_start:06d}.pt"
            self._flush(shard_name)
            self._shard_names.append(shard_name)

        index = {
            "num_samples": self._global_idx,
            "shard_size": self.shard_size,
            "num_shards": len(self._shard_names),
            "image_width": self._image_width,
            "image_height": self._image_height,
            "shards": self._shard_names,
        }
        index_path = os.path.join(self.shard_dir, "shard_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        print(f"  Shard index: {index_path}")

    def _flush(self, shard_name: str) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Packing requires torch. Install torch or disable --pack.") from exc

        shard_path = os.path.join(self.shard_dir, shard_name)
        torch.save(
            {
                "inputs": list(self._input_buf),
                "outputs": list(self._output_buf),
                "metadata": list(self._meta_buf),
            },
            shard_path,
        )
        print(f"  Wrote shard: {shard_path} ({len(self._meta_buf)} samples)")
        self._input_buf.clear()
        self._output_buf.clear()
        self._meta_buf.clear()
