"""Generate a balanced eval split for long multiplication.

Produces a fresh, disjoint-from-training eval set: brand-new random numbers
(never sampled from the training metadata.csv), balanced across all
(num_digits_a, num_digits_b) digit-length combos.

Exactly `--num_samples` tests are emitted. Since the number of combos may not
divide `--num_samples` evenly, the split is as balanced as possible: each combo
gets floor(num_samples / num_combos), and the remainder is spread one-per-combo
over the first `r` combos. For 100 samples over 9 combos this is 11/11/12 = 100.

Unlike `generate_random_digits_dataset.py`, which draws each sample's digit
length uniformly at random (unbalanced), this script stratifies by combo so that
every (num_digits_a, num_digits_b) combination is equally represented.

Emits PNG mode (individual images under images/input, images/output) with a
`metadata.csv` whose columns match the training metadata exactly. Set
`image` -> output (answer) and `edit_image` -> input (condition), matching the
PNG-mode semantics in `src/image_only/data.py`.
"""

import argparse
import csv
import os
import random
from dataclasses import replace

from config import DEFAULT_FONT_NAME, ColumnMulConfig, compute_max_digits_size, parse_image_size_arg
from generator import generate_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced eval split with all (num_digits_a, num_digits_b) "
            "combos equally represented."
        )
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Total number of eval samples")
    parser.add_argument("--min_digits", type=int, default=5, help="Minimum digits for a and b")
    parser.add_argument("--max_digits", type=int, default=7, help="Maximum digits for a and b")
    parser.add_argument("--base", type=int, default=10, help="Number base (default: 10)")
    parser.add_argument("--cell_size", type=int, default=16, help="Grid cell size in pixels")
    parser.add_argument("--image_width", type=str, default="pad",
                        help="Output image width. Integer (multiple of 16) or 'pad'.")
    parser.add_argument("--image_height", type=str, default="pad",
                        help="Output image height. Integer (multiple of 16) or 'pad'.")
    parser.add_argument("--font_name", type=str, default=DEFAULT_FONT_NAME, help="Monospace font name")
    parser.add_argument("--padding", type=int, default=16, help="Image padding in pixels")
    parser.add_argument("--line_width", type=int, default=None, help="Horizontal line width in pixels")
    parser.add_argument("--output_dir", type=str, default="output_balanced_eval", help="Output directory")
    parser.add_argument("--seed", type=int, default=1201, help="Random seed")
    parser.add_argument(
        "--exclude_metadata_csv",
        type=str,
        default=None,
        help=(
            "Path to a metadata.csv whose (a, b) pairs must not appear in the eval "
            "split (e.g. the training metadata.csv). Samples colliding with it are "
            "skipped and regenerated, keeping the split disjoint from training."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_digits > args.max_digits:
        raise SystemExit("min_digits must be <= max_digits")

    pad_w, pad_h = compute_max_digits_size(args.max_digits, args.max_digits, args.cell_size, args.padding)
    image_width = parse_image_size_arg(args.image_width, pad_w, "image_width")
    image_height = parse_image_size_arg(args.image_height, pad_h, "image_height")

    base_config = ColumnMulConfig(
        num_digits_a=args.min_digits,
        num_digits_b=args.min_digits,
        base=args.base,
        cell_size=args.cell_size,
        image_width=image_width,
        image_height=image_height,
        font_name=args.font_name,
        padding=args.padding,
        line_width=args.line_width,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    if base_config.line_width is None:
        base_config.line_width = max(base_config.cell_size // 16, 1)

    # Optional disjointness guard: load training (a, b) pairs to reject collisions.
    excluded_pairs = set()
    if args.exclude_metadata_csv:
        if not os.path.exists(args.exclude_metadata_csv):
            raise SystemExit(f"exclude_metadata_csv not found: {args.exclude_metadata_csv}")
        with open(args.exclude_metadata_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                excluded_pairs.add((row["a"], row["b"]))
        print(f"Loaded {len(excluded_pairs)} excluded (a, b) pairs from {args.exclude_metadata_csv}")

    # All digit-length combos in scan order.
    combos = [
        (a, b)
        for a in range(args.min_digits, args.max_digits + 1)
        for b in range(args.min_digits, args.max_digits + 1)
    ]
    num_combos = len(combos)
    per_combo, remainder = divmod(args.num_samples, num_combos)
    if per_combo < 1:
        raise SystemExit(
            f"num_samples={args.num_samples} too small for {num_combos} combos "
            f"(need at least one sample per combo)"
        )
    counts = [per_combo + (1 if i < remainder else 0) for i in range(num_combos)]
    assert sum(counts) == args.num_samples, (sum(counts), args.num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    examples_dir = os.path.join(args.output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    input_dir = os.path.join(images_dir, "input")
    output_dir = os.path.join(images_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    def rel_path(path: str) -> str:
        return os.path.relpath(path, args.output_dir).replace(os.sep, "/")

    metadata_list = []
    global_idx = 0
    combo_summary = []

    for combo_idx, (digits_a, digits_b) in enumerate(combos):
        # Stratified per combo: independent RNG stream so balance is exact.
        combo_rng = random.Random(args.seed + combo_idx)
        combo_config = replace(
            base_config, num_digits_a=digits_a, num_digits_b=digits_b
        )
        for _ in range(counts[combo_idx]):
            # Regenerate until the (a, b) pair is disjoint from the training split.
            while True:
                sample = generate_sample(combo_config, combo_rng)
                if (str(sample["a"]), str(sample["b"])) not in excluded_pairs:
                    break
            file_name = f"{global_idx:06d}.png"
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            sample["input_image"].save(input_path)
            sample["output_image"].save(output_path)

            metadata_list.append({
                "image": rel_path(output_path),
                "edit_image": rel_path(input_path),
                "prompt": "",
                "a": sample["a"],
                "b": sample["b"],
                "result": sample["result"],
                "num_digits_a": sample["num_digits_a"],
                "num_digits_b": sample["num_digits_b"],
                "base": sample["base"],
            })
            if global_idx == 0:
                sample["input_image"].save(os.path.join(examples_dir, "000000_input.png"))
                sample["output_image"].save(os.path.join(examples_dir, "000000_output.png"))
            global_idx += 1

        combo_summary.append((digits_a, digits_b, counts[combo_idx]))

    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    fieldnames = [
        "image", "edit_image", "prompt",
        "a", "b", "result", "num_digits_a", "num_digits_b", "base",
    ]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    print(f"Eval split saved to {args.output_dir}")
    print(f"  Total samples: {len(metadata_list)}")
    print(f"  Metadata file: {metadata_path}")
    print(f"  Example images: {examples_dir}/")
    print("  Combo distribution (num_digits_a, num_digits_b) -> count:")
    for digits_a, digits_b, count in combo_summary:
        print(f"    ({digits_a}, {digits_b}) -> {count}")


if __name__ == "__main__":
    main()