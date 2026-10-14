import csv
import os
import random
from typing import Dict, Any

from config import ColumnMulConfig
from arithmetic import compute_multiplication
from layout import create_layout
from renderer import render_image
from shard_writer import ShardWriter


def _generate_number(num_digits: int, base: int, rng: random.Random) -> int:
    lo = base ** (num_digits - 1)
    hi = base**num_digits - 1
    return rng.randint(lo, hi)


def generate_sample(
    config: ColumnMulConfig, rng: random.Random
) -> Dict[str, Any]:
    a = _generate_number(config.num_digits_a, config.base, rng)
    b = _generate_number(config.num_digits_b, config.base, rng)

    result = compute_multiplication(a, b, config.base)

    output_layout = create_layout(result, config, input_only=False)
    input_layout = create_layout(result, config, input_only=True)

    input_image = render_image(input_layout, config)
    output_image = render_image(output_layout, config)

    return {
        "a": a,
        "b": b,
        "result": result.result,
        "num_digits_a": config.num_digits_a,
        "num_digits_b": config.num_digits_b,
        "base": config.base,
        "input_image": input_image,
        "output_image": output_image,
    }


def generate_dataset(config: ColumnMulConfig, pack: bool = False, shard_size: int = 10000) -> None:
    if config.line_width is None:
        config.line_width = max(config.cell_size // 16, 1)

    rng = random.Random(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    examples_dir = os.path.join(config.output_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    shard_writer = ShardWriter(config.output_dir, shard_size) if pack else None

    if not pack:
        images_dir = os.path.join(config.output_dir, "images")
        input_dir = os.path.join(images_dir, "input")
        output_dir = os.path.join(images_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    metadata_list = []

    def _rel_path(path: str) -> str:
        return os.path.relpath(path, config.output_dir).replace(os.sep, "/")

    for i in range(config.num_samples):
        sample = generate_sample(config, rng)

        if pack:
            shard_url = shard_writer.add(
                sample["input_image"], sample["output_image"],
                {"a": sample["a"], "b": sample["b"], "result": sample["result"],
                 "num_digits_a": sample["num_digits_a"], "num_digits_b": sample["num_digits_b"],
                 "base": sample["base"]},
            )
            meta = {
                "image": shard_url,
                "edit_image": shard_url,
                "prompt": "",
                "a": sample["a"],
                "b": sample["b"],
                "result": sample["result"],
                "num_digits_a": sample["num_digits_a"],
                "num_digits_b": sample["num_digits_b"],
                "base": sample["base"],
            }
        else:
            file_name = f"{i:06d}.png"
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            sample["input_image"].save(input_path)
            sample["output_image"].save(output_path)

            meta = {
                "image": _rel_path(output_path),
                "edit_image": _rel_path(input_path),
                "prompt": "",
                "a": sample["a"],
                "b": sample["b"],
                "result": sample["result"],
                "num_digits_a": sample["num_digits_a"],
                "num_digits_b": sample["num_digits_b"],
                "base": sample["base"],
            }

        metadata_list.append(meta)

        if i == 0:
            sample["input_image"].save(os.path.join(examples_dir, "000000_input.png"))
            sample["output_image"].save(os.path.join(examples_dir, "000000_output.png"))

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{config.num_samples} samples")

    if pack:
        shard_writer.finalize()

    metadata_path = os.path.join(config.output_dir, "metadata.csv")
    fieldnames = ["image", "edit_image", "prompt", "a", "b", "result", "num_digits_a", "num_digits_b", "base"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_list)

    print(f"Dataset saved to {config.output_dir}")
    print(f"  Total samples: {config.num_samples}")
    print(f"  Metadata file: {metadata_path}")
    print(f"  Example images: {examples_dir}/")
