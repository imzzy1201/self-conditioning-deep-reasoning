"""Generate the A5 line-tracing image-conditioned dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from a5_line_trace_image_dataset.a5 import ALL_A5, Permutation, prefix_compositions, sequence_to_json
from a5_line_trace_image_dataset.shard_writer import ShardWriter

try:
    from .renderer import (
        DEFAULT_PALETTE,
        active_columns_for_steps,
        render_colored_trace,
        render_condition_trace,
        trace_paths,
    )
except ImportError:
    from renderer import (
        DEFAULT_PALETTE,
        active_columns_for_steps,
        render_colored_trace,
        render_condition_trace,
        trace_paths,
    )


FIELDNAMES = [
    "image",
    "edit_image",
    "prompt",
    "sequence",
    "paths",
    "prefixes",
    "patch_size",
    "image_width",
    "image_height",
    "canvas_columns",
    "active_columns",
    "column_stride",
    "num_permutations",
    "cot_depth",
    "line_width",
    "horizontal_line_width",
    "displacement3_line_width",
    "node_diameter",
    "palette",
    "split",
    "index",
]


def parse_splits(value: str) -> list[str]:
    splits = [item.strip() for item in value.split(",") if item.strip()]
    if not splits:
        raise argparse.ArgumentTypeError("Expected at least one split")
    return splits


def parse_depth_values(value: str) -> list[int]:
    depths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not depths:
        raise argparse.ArgumentTypeError("Expected at least one depth value")
    if any(depth <= 0 for depth in depths):
        raise argparse.ArgumentTypeError("All depth values must be positive")
    return depths


def parse_num_permutations_range(value: str) -> tuple[int, int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected a range in the form MIN,MAX")
    try:
        minimum, maximum = (int(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Range bounds must be integers") from exc
    if minimum <= 0 or maximum <= 0:
        raise argparse.ArgumentTypeError("Range bounds must be positive")
    if minimum > maximum:
        raise argparse.ArgumentTypeError("Range minimum cannot exceed maximum")
    return minimum, maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A5 permutation line-trace images")
    parser.add_argument("--splits", type=str, default="train,val,test_random,test_depth")
    parser.add_argument("--num_samples_per_split", type=int, default=1000)
    parser.add_argument("--num_train_samples", type=int, default=1000000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--image_width", type=int, default=1024)
    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument("--num_permutations", type=str, default="auto")
    length_group.add_argument(
        "--num_permutations_range",
        type=parse_num_permutations_range,
        default=None,
        metavar="MIN,MAX",
        help="Sample each non-test_depth input length uniformly from the inclusive integer range.",
    )
    parser.add_argument("--column_stride", type=int, default=4)
    parser.add_argument("--depth_values", type=str, default="4,8,12,15")
    parser.add_argument("--line_width", type=int, default=4, help="Width for non-horizontal line segments")
    parser.add_argument("--horizontal_line_width", type=int, default=3)
    parser.add_argument(
        "--displacement3_line_width",
        type=int,
        default=3,
        help="Width for line segments with abs(right_row - left_row) == 3.",
    )
    parser.add_argument("--node_diameter", type=int, default=9)
    parser.add_argument(
        "--adaptive_canvas",
        action="store_true",
        help=(
            "Size each sample's image width to exactly fit its trace, so shorter "
            "traces produce narrower images with no trailing white space. "
            "--image_width becomes the max cap (must still fit the largest length); "
            "the per-sample width is active_columns * patch_size."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="data/a5_line_trace_image/full_data_shard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pack", action="store_true")
    parser.add_argument("--shard_size", type=int, default=10000)
    return parser.parse_args()


def rel_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def palette_json() -> str:
    return json.dumps({str(k): list(v) for k, v in DEFAULT_PALETTE.items()}, separators=(",", ":"))


def paths_json(paths: list[list[int]]) -> str:
    return json.dumps(paths, separators=(",", ":"))


def choose_sequence(rng: random.Random, length: int) -> list[Permutation]:
    return [rng.choice(ALL_A5) for _ in range(length)]


def resolve_num_permutations(raw: str, max_steps: int) -> int:
    if raw == "auto":
        return max_steps
    value = int(raw)
    if value <= 0:
        raise SystemExit("--num_permutations must be positive")
    if value > max_steps:
        raise SystemExit("--num_permutations cannot exceed (image_width / patch_size - 1) / column_stride")
    return value


def samples_for_split(split: str, args: argparse.Namespace) -> int:
    if args.max_samples is not None:
        return args.max_samples
    if split == "train" and args.num_train_samples is not None:
        return args.num_train_samples
    return args.num_samples_per_split


def write_metadata(split_dir: str, rows: list[dict[str, Any]]) -> None:
    metadata_path = os.path.join(split_dir, "metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Metadata: {metadata_path}")


def export_split(
    split: str,
    split_idx: int,
    args: argparse.Namespace,
    num_samples: int,
    canvas_columns: int,
    default_num_permutations: int | None,
    num_permutations_range: tuple[int, int] | None,
    depth_values: list[int],
) -> None:
    split_dir = os.path.join(args.output_dir, split)
    examples_dir = os.path.join(split_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    input_dir = os.path.join(split_dir, "images", "input")
    output_dir = os.path.join(split_dir, "images", "output")
    if not args.pack:
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    shard_writer = ShardWriter(split_dir, args.shard_size) if args.pack else None
    image_height = 5 * args.patch_size
    rows: list[dict[str, Any]] = []
    pal_json = palette_json()

    rng = random.Random(args.seed + split_idx * 1000003)
    sample_iter = range(num_samples)
    if tqdm is not None:
        sample_iter = tqdm(sample_iter, total=num_samples, desc=f"[{split}] generating", unit="sample")

    for idx in sample_iter:
        if split == "test_depth":
            num_permutations = depth_values[idx % len(depth_values)]
            if active_columns_for_steps(num_permutations, args.column_stride) > canvas_columns:
                raise SystemExit(
                    f"Depth {num_permutations} needs {active_columns_for_steps(num_permutations, args.column_stride)} "
                    f"columns but canvas has {canvas_columns}"
                )
        elif num_permutations_range is not None:
            # randint is inclusive at both ends and samples each integer uniformly.
            num_permutations = rng.randint(*num_permutations_range)
        else:
            if default_num_permutations is None:
                raise RuntimeError("A fixed permutation length is required when no range is configured")
            num_permutations = default_num_permutations

        sequence = choose_sequence(rng, num_permutations)
        paths = trace_paths(sequence)
        prefixes = prefix_compositions(sequence)
        active_cols = active_columns_for_steps(num_permutations, args.column_stride)
        if args.adaptive_canvas:
            # Canvas exactly fits this trace: no trailing white space.
            sample_image_width = active_cols * args.patch_size
            sample_canvas_columns = active_cols
        else:
            sample_image_width = args.image_width
            sample_canvas_columns = canvas_columns
        condition = render_condition_trace(
            sequence,
            patch_size=args.patch_size,
            image_width=sample_image_width,
            column_stride=args.column_stride,
            line_width=args.line_width,
            horizontal_line_width=args.horizontal_line_width,
            displacement3_line_width=args.displacement3_line_width,
            node_diameter=args.node_diameter,
        )
        target = render_colored_trace(
            sequence,
            patch_size=args.patch_size,
            image_width=sample_image_width,
            column_stride=args.column_stride,
            line_width=args.line_width,
            horizontal_line_width=args.horizontal_line_width,
            displacement3_line_width=args.displacement3_line_width,
            node_diameter=args.node_diameter,
        )

        if args.pack:
            if shard_writer is None:
                raise RuntimeError("Shard writer is not initialized")
            image_ref = shard_writer.add(
                condition_image=condition,
                target_image=target,
                metadata={
                    "sequence": [list(perm) for perm in sequence],
                    "paths": paths,
                    "prefixes": [list(perm) for perm in prefixes],
                    "split": split,
                    "index": idx,
                },
            )
            edit_image_ref = image_ref
        else:
            filename = f"{idx:06d}.png"
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            condition.save(input_path)
            target.save(output_path)
            edit_image_ref = rel_path(input_path, split_dir)
            image_ref = rel_path(output_path, split_dir)

        if idx < 3:
            condition.save(os.path.join(examples_dir, f"{idx:06d}_input.png"))
            target.save(os.path.join(examples_dir, f"{idx:06d}_output.png"))

        rows.append(
            {
                "image": image_ref,
                "edit_image": edit_image_ref,
                "prompt": "",
                "sequence": sequence_to_json(sequence),
                "paths": paths_json(paths),
                "prefixes": sequence_to_json(prefixes),
                "patch_size": args.patch_size,
                "image_width": sample_image_width,
                "image_height": image_height,
                "canvas_columns": sample_canvas_columns,
                "active_columns": active_cols,
                "column_stride": args.column_stride,
                "num_permutations": num_permutations,
                "cot_depth": num_permutations,
                "line_width": args.line_width,
                "horizontal_line_width": args.horizontal_line_width,
                "displacement3_line_width": args.displacement3_line_width,
                "node_diameter": args.node_diameter,
                "palette": pal_json,
                "split": split,
                "index": idx,
            }
        )

    if shard_writer is not None:
        shard_writer.finalize()
    write_metadata(split_dir, rows)
    print(f"Split '{split}' saved to {split_dir} ({len(rows)} samples)")


def main() -> None:
    args = parse_args()
    splits = parse_splits(args.splits)
    depth_values = parse_depth_values(args.depth_values)

    if args.num_samples_per_split <= 0:
        raise SystemExit("--num_samples_per_split must be positive")
    if args.num_train_samples is not None and args.num_train_samples <= 0:
        raise SystemExit("--num_train_samples must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max_samples must be positive")
    if args.patch_size <= 0:
        raise SystemExit("--patch_size must be positive")
    if args.image_width <= 0 or args.image_width % args.patch_size != 0:
        raise SystemExit("--image_width must be a positive multiple of --patch_size")
    if args.column_stride <= 0:
        raise SystemExit("--column_stride must be positive")
    if args.line_width <= 0:
        raise SystemExit("--line_width must be positive")
    if args.horizontal_line_width <= 0:
        raise SystemExit("--horizontal_line_width must be positive")
    if args.displacement3_line_width <= 0:
        raise SystemExit("--displacement3_line_width must be positive")
    if args.node_diameter <= 0:
        raise SystemExit("--node_diameter must be positive")
    if args.image_width % 16 != 0 or (5 * args.patch_size) % 16 != 0:
        raise SystemExit("image width and height must be multiples of 16")

    canvas_columns = args.image_width // args.patch_size
    max_steps = (canvas_columns - 1) // args.column_stride
    if args.num_permutations_range is not None:
        _, range_max = args.num_permutations_range
        if range_max > max_steps:
            raise SystemExit(
                "--num_permutations_range cannot exceed "
                "(image_width / patch_size - 1) / column_stride"
            )
        default_num_permutations = None
    else:
        default_num_permutations = resolve_num_permutations(args.num_permutations, max_steps)
    os.makedirs(args.output_dir, exist_ok=True)

    print("A5 line-trace image-conditioned dataset generation")
    print(f"  output_dir: {args.output_dir}")
    print(f"  image_size: {args.image_width}x{5 * args.patch_size}")
    print(f"  canvas_columns: {canvas_columns}")
    print(f"  column_stride: {args.column_stride}")
    if args.num_permutations_range is not None:
        print(
            "  num_permutations_range: "
            f"{args.num_permutations_range[0]},{args.num_permutations_range[1]} (uniform, inclusive)"
        )
    else:
        print(f"  default_num_permutations: {default_num_permutations}")
        print(f"  active_columns: {active_columns_for_steps(default_num_permutations, args.column_stride)}")
    print(f"  adaptive_canvas: {args.adaptive_canvas}")
    print(f"  line_width: {args.line_width}")
    print(f"  horizontal_line_width: {args.horizontal_line_width}")
    print(f"  displacement3_line_width: {args.displacement3_line_width}")
    print(f"  node_diameter: {args.node_diameter}")
    print(f"  samples_per_split: {args.max_samples or args.num_samples_per_split}")
    if "train" in splits and args.num_train_samples is not None and args.max_samples is None:
        print(f"  train_samples: {args.num_train_samples}")
    print(f"  pack: {args.pack}")

    for split_idx, split in enumerate(splits):
        num_samples = samples_for_split(split, args)
        export_split(
            split=split,
            split_idx=split_idx,
            args=args,
            num_samples=num_samples,
            canvas_columns=canvas_columns,
            default_num_permutations=default_num_permutations,
            num_permutations_range=args.num_permutations_range,
            depth_values=depth_values,
        )

    print("All requested splits have been exported.")


if __name__ == "__main__":
    main()
