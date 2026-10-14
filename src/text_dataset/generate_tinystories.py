import argparse
import os


PROMPT_PREFIX = "Render these texts with a white background and black text: "


def extract_tinystories_prompt(text: str) -> dict[str, str]:
    """Extract prompt from TinyStories text column.

    prompt = "Render these texts..." prefix + full story text (all paragraphs).
    """
    full_text = text.strip()
    prompt = PROMPT_PREFIX + full_text if full_text else ""
    return {"prompt": prompt}


def parse_split_mapping(raw: str) -> dict[str, str]:
    mapping = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            raise SystemExit(f"Invalid split mapping pair: '{pair}'. Expected format: hf_split=output_split")
        hf_split, output_split = pair.split("=", 1)
        mapping[hf_split.strip()] = output_split.strip()
    if not mapping:
        raise SystemExit("--split_mapping must contain at least one mapping")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TinyStories to text image dataset"
    )

    # --- Dataset loading ---
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories",
                        help="HuggingFace dataset name")
    parser.add_argument("--splits", type=str, default="train,validation",
                        help="Comma-separated HF splits to process")
    parser.add_argument("--split_mapping", type=str, default="train=train,validation=test",
                        help="Maps HF split names to output split names (format: hf1=out1,hf2=out2)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per split (applied after materialization)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle each split before truncating")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle")
    parser.add_argument("--hf_endpoint", type=str, default="https://hf-mirror.com",
                        help="HuggingFace mirror endpoint")

    # --- Image rendering ---
    parser.add_argument("--image_width", type=int, default=512, help="Image width in pixels (must be multiple of 16)")
    parser.add_argument("--image_height", type=str, default="512",
                        help="Image height in pixels (multiple of 16), or 'pad' to auto-compute")
    parser.add_argument("--padding", type=int, default=16, help="Page padding in pixels")
    parser.add_argument("--font_name", type=str, default="Consolas", help="Font name or .ttf path")
    parser.add_argument("--font_size", type=int, default=13, help="Font size in pixels")
    parser.add_argument("--line_spacing", type=int, default=4, help="Extra spacing after each text line")
    parser.add_argument("--paragraph_gap_lines", type=int, default=1,
                        help="Number of blank lines between paragraphs")

    # --- Output ---
    parser.add_argument("--output_dir", type=str, default="output_tinystories", help="Output directory")
    parser.add_argument("--pack", action="store_true", help="Pack images into shard .pt files")
    parser.add_argument("--shard_size", type=int, default=10000, help="Samples per shard when --pack is enabled")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of parallel rendering workers (0 = sequential)")
    parser.add_argument("--no_edit_image", action="store_true",
                        help="Do not render prompt as condition image; store prompt as text only")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.image_width % 16 != 0:
        raise SystemExit("--image_width must be a multiple of 16")
    if args.padding < 0:
        raise SystemExit("--padding must be >= 0")
    if args.font_size <= 0:
        raise SystemExit("--font_size must be positive")
    if args.line_spacing < 0:
        raise SystemExit("--line_spacing must be >= 0")
    if args.paragraph_gap_lines < 0:
        raise SystemExit("--paragraph_gap_lines must be >= 0")
    if args.shard_size <= 0:
        raise SystemExit("--shard_size must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max_samples must be positive when set")
    if args.num_workers < 0:
        raise SystemExit("--num_workers must be >= 0")

    # Set HF endpoint before importing datasets library
    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    from datasets import load_dataset

    try:
        from .utils import parse_image_height_arg, parse_splits
        from .convert_dataset import convert_dataset_to_images
    except ImportError:
        from utils import parse_image_height_arg, parse_splits
        from convert_dataset import convert_dataset_to_images

    image_height = parse_image_height_arg(args.image_height)
    if image_height is None:
        # pad mode not supported with truncation-based rendering; use fixed 512
        raise SystemExit("--image_height 'pad' mode is not supported for text datasets with truncation. "
                         "Specify a fixed height (e.g. 512).")

    split_mapping = parse_split_mapping(args.split_mapping)
    splits = parse_splits(args.splits)

    os.makedirs(args.output_dir, exist_ok=True)

    for hf_split in splits:
        output_split = split_mapping.get(hf_split, hf_split)

        print(f"Loading split '{hf_split}' from {args.dataset_name} (output as '{output_split}')...")
        dataset = load_dataset(args.dataset_name, split=hf_split)

        if args.shuffle:
            dataset = dataset.shuffle(seed=args.seed)

        if args.max_samples is not None:
            num = min(args.max_samples, len(dataset))
            dataset = dataset.select(range(num))

        dataset = dataset.map(lambda row: extract_tinystories_prompt(row["text"]))

        convert_dataset_to_images(
            dataset=dataset,
            output_dir=args.output_dir,
            split=output_split,
            prompt_prefix=PROMPT_PREFIX,
            image_width=args.image_width,
            image_height=image_height,
            padding=args.padding,
            font_name=args.font_name,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
            paragraph_gap_lines=args.paragraph_gap_lines,
            pack=args.pack,
            shard_size=args.shard_size,
            num_workers=args.num_workers,
            no_edit_image=args.no_edit_image,
        )

    print("All requested splits have been exported.")


if __name__ == "__main__":
    main()