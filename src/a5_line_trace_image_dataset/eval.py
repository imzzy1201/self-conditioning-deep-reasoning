"""Evaluate A5 line-trace generated images by sampling expected colored paths."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
from statistics import mean, median
from typing import Any

import numpy as np
from PIL import Image

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from a5_line_trace_image_dataset.renderer import render_colored_trace


SHARD_PREFIX = "shard:"
SAMPLE_TS = (0.20, 0.35, 0.65, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A5 line-trace images")
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_dir", type=str, help="Directory with generated 000000.png files")
    group.add_argument("--pred_root", type=str, help="Score metadata image refs relative to this root")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--strict_size", action="store_true", default=True)
    parser.add_argument("--no_strict_size", dest="strict_size", action="store_false")
    return parser.parse_args()


def load_rows(metadata_csv: str) -> list[dict[str, str]]:
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("metadata.csv is empty")
    return rows


def parse_palette(value: str) -> dict[int, np.ndarray]:
    raw = json.loads(value)
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def parse_paths(value: str) -> list[list[int]]:
    return [[int(row) for row in path] for path in json.loads(value)]


def parse_sequence(value: str) -> list[tuple[int, int, int, int, int]]:
    return [tuple(int(item) for item in perm) for perm in json.loads(value)]  # type: ignore[list-item]


def parse_shard_url(url: str) -> tuple[int, int]:
    parts = url[len(SHARD_PREFIX):].split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid shard URL: {url}")
    return int(parts[0]), int(parts[1])


def resolve_path(path_str: str, root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else root / path


class ShardReader:
    def __init__(self, root: Path):
        self.root = root
        self.shard_dir = root / "shards"
        self.cache: dict[int, dict[str, Any]] = {}

    def load_image(self, url: str, key: str = "outputs") -> Image.Image:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Reading shard predictions requires torch") from exc

        shard_start, local_idx = parse_shard_url(url)
        if shard_start not in self.cache:
            shard_path = self.shard_dir / f"shard_{shard_start:06d}.pt"
            self.cache[shard_start] = torch.load(shard_path, weights_only=False)
        entry = self.cache[shard_start][key][local_idx]
        if isinstance(entry, bytes):
            return Image.open(io.BytesIO(entry)).convert("RGB")
        arr = entry.permute(1, 2, 0).numpy().astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")


def load_image_ref(ref: str, root: Path, shard_reader: ShardReader) -> Image.Image:
    if ref.startswith(SHARD_PREFIX):
        return shard_reader.load_image(ref, "outputs")
    path = resolve_path(ref, root)
    with Image.open(path) as img:
        return img.convert("RGB")


def prediction_image_for_row(
    row: dict[str, str],
    row_pos: int,
    args: argparse.Namespace,
    data_root: Path,
) -> Image.Image:
    if args.pred_dir:
        idx = int(row.get("index", row_pos))
        path = Path(args.pred_dir) / f"{idx:06d}.png"
        if not path.exists():
            alt = Path(args.pred_dir) / f"{row_pos:06d}.png"
            legacy_matches = sorted(Path(args.pred_dir).glob(f"{row_pos:05d}_*.png"))
            if alt.exists():
                path = alt
            elif legacy_matches:
                path = legacy_matches[0]
            else:
                raise FileNotFoundError(f"Missing prediction image: {path}")
        with Image.open(path) as img:
            return img.convert("RGB")

    pred_root = Path(args.pred_root)
    shard_reader = ShardReader(pred_root)
    return load_image_ref(row["image"], pred_root if args.pred_root else data_root, shard_reader)


def point_for_cell(row: int, col: int, patch_size: int) -> tuple[float, float]:
    return ((col - 0.5) * patch_size, (row - 0.5) * patch_size)


def nearest_palette_color(rgb: np.ndarray, palette: dict[int, np.ndarray]) -> int:
    return min(palette, key=lambda key: float(np.linalg.norm(rgb - palette[key])))


def sample_rgb(arr: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    h, w, _ = arr.shape
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    return arr[y0:y1, x0:x1].mean(axis=(0, 1))


def score_segment(
    arr: np.ndarray,
    expected_arr: np.ndarray,
    left_row: int,
    right_row: int,
    step: int,
    patch_size: int,
    column_stride: int,
    radius: int,
    tolerance: float,
) -> bool:
    x0, y0 = point_for_cell(left_row, 1 + column_stride * (step - 1), patch_size)
    x1, y1 = point_for_cell(right_row, 1 + column_stride * step, patch_size)
    votes = 0
    for t in SAMPLE_TS:
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        rgb = sample_rgb(arr, x, y, radius)
        expected_rgb = sample_rgb(expected_arr, x, y, radius)
        if float(np.linalg.norm(rgb - expected_rgb)) <= tolerance:
            votes += 1
    return votes >= (len(SAMPLE_TS) // 2 + 1)


def score_sample(row: dict[str, str], image: Image.Image, args: argparse.Namespace) -> dict[str, Any]:
    patch_size = int(row["patch_size"])
    num_permutations = int(row["num_permutations"])
    expected_width = int(row["image_width"])
    expected_height = int(row["image_height"])
    if args.strict_size and image.size != (expected_width, expected_height):
        raise ValueError(f"Expected image size {(expected_width, expected_height)}, got {image.size}")
    if image.size != (expected_width, expected_height):
        image = image.resize((expected_width, expected_height), Image.Resampling.BICUBIC)

    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    paths = parse_paths(row["paths"])
    column_stride = int(row.get("column_stride", "4"))
    line_width = int(row.get("line_width", "4"))
    horizontal_line_width = int(row.get("horizontal_line_width", "3"))
    displacement3_line_width = int(row.get("displacement3_line_width", str(line_width)))
    node_diameter = int(row.get("node_diameter", "9"))
    radius = max(1, max(line_width, horizontal_line_width, displacement3_line_width) // 3)
    tolerance = 40.0
    expected = render_colored_trace(
        parse_sequence(row["sequence"]),
        patch_size=patch_size,
        image_width=expected_width,
        column_stride=column_stride,
        line_width=line_width,
        horizontal_line_width=horizontal_line_width,
        displacement3_line_width=displacement3_line_width,
        node_diameter=node_diameter,
    )
    expected_arr = np.asarray(expected.convert("RGB"), dtype=np.float32)

    segment_correct: list[bool] = []
    step_correct: list[bool] = []
    for step in range(1, num_permutations + 1):
        current_step = []
        for start_idx, path in enumerate(paths, start=1):
            ok = score_segment(
                arr=arr,
                expected_arr=expected_arr,
                left_row=path[step - 1],
                right_row=path[step],
                step=step,
                patch_size=patch_size,
                column_stride=column_stride,
                radius=radius,
                tolerance=tolerance,
            )
            segment_correct.append(ok)
            current_step.append(ok)
        step_correct.append(all(current_step))

    max_prefix = 0
    for ok in step_correct:
        if not ok:
            break
        max_prefix += 1

    return {
        "index": int(row.get("index", 0)),
        "split": row.get("split", ""),
        "cot_depth": int(row.get("cot_depth", num_permutations)),
        "num_permutations": num_permutations,
        "max_correct_prefix_steps": max_prefix,
        "first_error_step": max_prefix + 1 if max_prefix < num_permutations else num_permutations,
        "step_accuracy": sum(step_correct) / len(step_correct),
        "segment_accuracy": sum(segment_correct) / len(segment_correct),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    frac = pos - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def aggregate_depth(samples: list[dict[str, Any]]) -> dict[str, float]:
    max_steps = [float(item["max_correct_prefix_steps"]) for item in samples]
    return {
        "num_samples": float(len(samples)),
        "max_correct_prefix_steps.mean": mean(max_steps),
        "step_accuracy.mean": mean(float(item["step_accuracy"]) for item in samples),
        "segment_accuracy.mean": mean(float(item["segment_accuracy"]) for item in samples),
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    max_steps = [float(item["max_correct_prefix_steps"]) for item in samples]
    step_acc = [float(item["step_accuracy"]) for item in samples]
    seg_acc = [float(item["segment_accuracy"]) for item in samples]
    out: dict[str, Any] = {
        "num_samples": len(samples),
        "max_correct_prefix_steps.mean": mean(max_steps),
        "max_correct_prefix_steps.median": median(max_steps),
        "max_correct_prefix_steps.p10": percentile(max_steps, 0.10),
        "step_accuracy.mean": mean(step_acc),
        "segment_accuracy.mean": mean(seg_acc),
    }
    by_depth: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_depth.setdefault(int(sample["cot_depth"]), []).append(sample)
    out["by_cot_depth"] = {
        str(depth): aggregate_depth(depth_samples) for depth, depth_samples in sorted(by_depth.items())
    }
    return out


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metadata_csv)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples must be positive")
        rows = rows[:args.max_samples]
    data_root = Path(args.data_root) if args.data_root else Path(args.metadata_csv).parent

    samples: list[dict[str, Any]] = []
    for row_pos, row in enumerate(rows):
        image = prediction_image_for_row(row, row_pos, args, data_root)
        samples.append(score_sample(row, image, args))

    metrics = aggregate(samples)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if args.output_jsonl:
        os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, separators=(",", ":")) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
