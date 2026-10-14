"""Shard-based packing utility for dataset images.

Writes images as PNG-compressed bytes into shard .pt files instead of individual PNGs.
Each shard contains:
  - "inputs":  list[bytes] (PNG-encoded condition images)
  - "outputs": list[bytes] (PNG-encoded target images)
  - "metadata": list[dict] (per-sample CSV row data)
"""

import io
import json
import os

import torch


def pil_to_png_bytes(img) -> bytes:
    """Compress a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ShardWriter:
    """Accumulates samples and writes shard .pt files every shard_size samples."""

    def __init__(self, output_dir: str, shard_size: int = 10000):
        self.shard_dir = os.path.join(output_dir, "shards")
        self.shard_size = shard_size
        self.output_dir = output_dir
        os.makedirs(self.shard_dir, exist_ok=True)

        self._input_buf: list[bytes] = []
        self._output_buf: list[bytes] = []
        self._meta_buf: list[dict] = []
        self._global_idx = 0
        self._shard_names: list[str] = []
        self._image_height: int | None = None
        self._image_width: int | None = None

    def add(self, condition_image, target_image, metadata: dict) -> str:
        """Add one sample. Returns shard URL string (e.g. "shard:000000:5").

        condition_image: the input/problem image (stored in "inputs")
        target_image: the output/answer image (stored in "outputs")
        """
        condition_bytes = pil_to_png_bytes(condition_image)
        target_bytes = pil_to_png_bytes(target_image)

        w, h = condition_image.size
        if self._image_height is None:
            self._image_height = h
            self._image_width = w

        self._input_buf.append(condition_bytes)
        self._output_buf.append(target_bytes)
        self._meta_buf.append(metadata)

        shard_idx = self._global_idx // self.shard_size
        local_idx = self._global_idx % self.shard_size
        shard_name = f"shard_{shard_idx * self.shard_size:06d}.pt"
        shard_url = f"shard:{shard_idx * self.shard_size:06d}:{local_idx}"

        self._global_idx += 1

        if len(self._input_buf) == self.shard_size:
            self._flush(shard_name)
            self._shard_names.append(shard_name)

        return shard_url

    def finalize(self) -> None:
        """Write remaining samples (if any) and shard_index.json."""
        if len(self._input_buf) > 0:
            shard_idx = len(self._shard_names)
            shard_name = f"shard_{shard_idx * self.shard_size:06d}.pt"
            self._flush(shard_name)
            self._shard_names.append(shard_name)

        index = {
            "num_samples": self._global_idx,
            "shard_size": self.shard_size,
            "num_shards": len(self._shard_names),
            "image_height": self._image_height,
            "image_width": self._image_width,
            "shards": self._shard_names,
        }
        index_path = os.path.join(self.shard_dir, "shard_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(f"  Shard index: {index_path}")
        print(f"  Shard files: {len(self._shard_names)}")
        print(f"  Total samples packed: {self._global_idx}")

    def _flush(self, shard_name: str) -> None:
        inputs = list(self._input_buf)
        outputs = list(self._output_buf)
        metadata = list(self._meta_buf)

        shard_path = os.path.join(self.shard_dir, shard_name)
        torch.save({"inputs": inputs, "outputs": outputs, "metadata": metadata}, shard_path)
        print(f"  Wrote shard: {shard_path} ({len(metadata)} samples)")

        self._input_buf.clear()
        self._output_buf.clear()
        self._meta_buf.clear()