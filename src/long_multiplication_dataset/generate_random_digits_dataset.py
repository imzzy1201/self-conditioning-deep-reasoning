import argparse
import csv
import os
import random
from dataclasses import replace

from config import DEFAULT_FONT_NAME, ColumnMulConfig, compute_max_digits_size, parse_image_size_arg
from generator import generate_sample
from shard_writer import ShardWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset with random digit lengths per sample"
    )
    parser.add_argument("--min_digits", type=int, default=3, help="Minimum digits for a and b")
    parser.add_argument("--max_digits", type=int, default=6, help="Maximum digits for a and b")
    parser.add_argument("--base", type=int, default=10, help="Number base (default: 10)")
    parser.add_argument("--cell_size", type=int, default=16, help="Grid cell size in pixels")
    parser.add_argument("--image_width", type=str, default=None, help="Output image width. Integer (multiple of 16) or 'pad' to pad to max-digits size.")
    parser.add_argument("--image_height", type=str, default=None, help="Output image height. Integer (multiple of 16) or 'pad' to pad to max-digits size.")
    parser.add_argument("--font_name", type=str, default=DEFAULT_FONT_NAME, help="Monospace font name")
    parser.add_argument("--padding", type=int, default=16, help="Image padding in pixels")
    parser.add_argument("--line_width", type=int, default=None, help="Horizontal line width in pixels (default: auto-scale)")
    parser.add_argument("--num_samples", type=int, default=200000, help="Number of samples to generate")
    parser.add_argument("--balance_digits", action="store_true", help="Balance (num_digits_a, num_digits_b) combos evenly: each combo gets floor(num_samples / num_combos) samples, with the remainder distributed one-per-combo to the first combos. Recommended for test splits so per-digit accuracy is comparable across buckets. Off by default (i.i.d. uniform draws).")
    parser.add_argument("--output_dir", type=str, default="output_random_digits", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--pack", action="store_true", help="Pack images into shard .pt files instead of individual PNGs")
    parser.add_argument("--shard_size", type=int, default=10000, help="Number of samples per shard file (only used with --pack)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_digits > args.max_digits:
        raise SystemExit("min_digits must be <= max_digits")

    # Compute max-digits padded size (for 'pad' mode)
    pad_w, pad_h = None, None
    needs_pad = (args.image_width and args.image_width.lower() == "pad") or \
                (args.image_height and args.image_height.lower() == "pad")
    if needs_pad:
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
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    if base_config.line_width is None:
        base_config.line_width = max(base_config.cell_size // 16, 1)

    rng = random.Random(base_config.seed)

    os.makedirs(base_config.output_dir, exist_ok=True)
    examples_dir = os.path.join(base_config.output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    shard_writer = ShardWriter(base_config.output_dir, args.shard_size) if args.pack else None

    if not args.pack:
        images_dir = os.path.join(base_config.output_dir, "images")
        input_dir = os.path.join(images_dir, "input")
        output_dir = os.path.join(images_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    metadata_list = []

    def rel_path(path: str) -> str:
        return os.path.relpath(path, base_config.output_dir).replace(os.sep, "/")

    # Optional balanced quota across (num_digits_a, num_digits_b) combos so each
    # bucket has an equal count (useful for per-digit accuracy eval). When off,
    # digit counts are drawn i.i.d. uniform (the original behavior).
    if args.balance_digits:
        combos = [(da, db)
                  for da in range(args.min_digits, args.max_digits + 1)
                  for db in range(args.min_digits, args.max_digits + 1)]
        num_combos = len(combos)
        base_q = base_config.num_samples // num_combos
        extra = base_config.num_samples - base_q * num_combos
        plan = []
        for k, (da, db) in enumerate(combos):
            plan.extend([(da, db)] * (base_q + (1 if k < extra else 0)))
        rng.shuffle(plan)
        assert len(plan) == base_config.num_samples
        print(f"balance_digits: {num_combos} combos x ~{base_q}"
              f"{('+' + str(extra) + ' extra') if extra else ''} = {len(plan)} samples")

    for i in range(base_config.num_samples):
        if args.balance_digits:
            digits_a, digits_b = plan[i]
        else:
            digits_a = rng.randint(args.min_digits, args.max_digits)
            digits_b = rng.randint(args.min_digits, args.max_digits)
        sample_config = replace(base_config, num_digits_a=digits_a, num_digits_b=digits_b)
        sample = generate_sample(sample_config, rng)

        sample_meta = {
            "a": sample["a"],
            "b": sample["b"],
            "result": sample["result"],
            "num_digits_a": sample["num_digits_a"],
            "num_digits_b": sample["num_digits_b"],
            "base": sample["base"],
        }

        if args.pack:
            shard_url = shard_writer.add(sample["input_image"], sample["output_image"], sample_meta)
            meta = {
                "image": shard_url,
                "edit_image": shard_url,
                "prompt": "",
                **sample_meta,
            }
        else:
            file_name = f"{i:06d}.png"
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            sample["input_image"].save(input_path)
            sample["output_image"].save(output_path)

            meta = {
                "image": rel_path(output_path),
                "edit_image": rel_path(input_path),
                "prompt": "",
                **sample_meta,
            }

        metadata_list.append(meta)

        if i == 0:
            sample["input_image"].save(os.path.join(examples_dir, "000000_input.png"))
            sample["output_image"].save(os.path.join(examples_dir, "000000_output.png"))

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{base_config.num_samples} samples")

    if args.pack:
        shard_writer.finalize()

    metadata_path = os.path.join(base_config.output_dir, "metadata.csv")
    fieldnames = [
        "image", "edit_image", "prompt",
        "a", "b", "result", "num_digits_a", "num_digits_b", "base",
    ]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    print(f"Dataset saved to {base_config.output_dir}")
    print(f"  Total samples: {base_config.num_samples}")
    print(f"  Metadata file: {metadata_path}")
    print(f"  Example images: {examples_dir}/")


if __name__ == "__main__":
    main()
