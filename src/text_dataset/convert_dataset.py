import concurrent.futures
import csv
import io
import multiprocessing
import os
from typing import Any

from datasets import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from .rendering import load_font, measure_text_height, render_text_image
    from .shard_writer import ShardWriter
    from .utils import rel_path
    from ._worker import render_sample, render_sample_no_edit
except ImportError:
    from rendering import load_font, measure_text_height, render_text_image
    from shard_writer import ShardWriter
    from utils import rel_path
    from _worker import render_sample, render_sample_no_edit


def _effective_cpu_count() -> int:
    """Return the number of CPUs actually available, respecting k8s cgroup limits."""
    # cgroups v2 (modern k8s)
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
            if parts[0] != "max":
                return max(1, int(parts[0]) // int(parts[1]))
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    # cgroups v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
        if quota > 0:
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            return max(1, quota // period)
    except (FileNotFoundError, ValueError, PermissionError):
        pass
    return os.cpu_count() or 1


def _cgroup_memory_gb() -> float:
    """Return current cgroup memory usage in GB (for k8s OOM monitoring)."""
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read().strip()) / (1024 ** 3)
    except (FileNotFoundError, PermissionError):
        return -1.0


def _process_rendered_sample(
    idx: int,
    input_image: Any,
    output_image: Any,
    prompt_text: str,
    split: str,
    pack: bool,
    shard_writer: Any,
    split_dir: str,
    input_dir: str,
    output_png_dir: str,
    examples_dir: str,
    metadata_rows: list[dict[str, Any]],
    no_edit_image: bool = False,
) -> None:
    """Post-process a rendered sample: shard/file writing, example saving, metadata."""
    if pack:
        if shard_writer is None:
            raise RuntimeError("Shard writer is not initialized")
        shard_url = shard_writer.add(
            condition_image=input_image if input_image is not None else output_image,  # placeholder for no_edit_image
            target_image=output_image,
            metadata={
                "prompt": prompt_text,
                "split": split,
                "index": idx,
            },
        )
        image_ref = shard_url
        edit_image_ref = "" if no_edit_image else shard_url
    else:
        filename = f"{idx:06d}.png"
        output_path = os.path.join(output_png_dir, filename)
        output_image.save(output_path)
        image_ref = rel_path(output_path, split_dir)

        if no_edit_image:
            edit_image_ref = ""
        else:
            input_path = os.path.join(input_dir, filename)
            input_image.save(input_path)
            edit_image_ref = rel_path(input_path, split_dir)

    if idx < 3:
        if not no_edit_image and input_image is not None:
            input_image.save(os.path.join(examples_dir, f"{idx:06d}_input.png"))
        output_image.save(os.path.join(examples_dir, f"{idx:06d}_output.png"))

    metadata_rows.append(
        {
            "image": image_ref,
            "edit_image": edit_image_ref,
            "prompt": prompt_text,
            "split": split,
            "index": idx,
        }
    )


def convert_dataset_to_images(
    dataset: Dataset,
    output_dir: str,
    split: str,
    prompt_column: str = "prompt",
    prompt_prefix: str = "",
    image_width: int = 512,
    image_height: int = 512,
    padding: int = 16,
    font_name: str = "Consolas",
    font_size: int = 13,
    line_spacing: int = 4,
    paragraph_gap_lines: int = 1,
    pack: bool = True,
    shard_size: int = 10000,
    num_workers: int = 0,
    no_edit_image: bool = False,
) -> None:
    """Convert a text Dataset into a sharded image dataset.

    Each sample produces one or two images:
    - output image: renders the text from prompt (stripped of prompt_prefix)
    - input image: renders the prompt text (skipped when no_edit_image=True)

    When prompt_prefix is set, the output image text is prompt_text minus the
    prefix (e.g. "Render these texts..." is stripped before rendering).

    When no_edit_image=True, the prompt is stored as text in metadata.csv
    and edit_image is left empty — useful for text-to-image training data.

    Text is auto-wrapped; lines exceeding the image height are truncated.
    """
    if image_width % 16 != 0:
        raise SystemExit("--image_width must be a multiple of 16")
    if image_height % 16 != 0:
        raise SystemExit("--image_height must be a multiple of 16")

    first = dataset[0]
    if prompt_column not in first:
        raise SystemExit(f"Dataset is missing required column: {prompt_column}")

    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    examples_dir = os.path.join(split_dir, "examples")
    os.makedirs(examples_dir, exist_ok=True)

    shard_writer = ShardWriter(split_dir, shard_size) if pack else None

    input_dir = ""
    output_png_dir = ""
    if not pack:
        output_png_dir = os.path.join(split_dir, "images", "output")
        os.makedirs(output_png_dir, exist_ok=True)
        if not no_edit_image:
            input_dir = os.path.join(split_dir, "images", "input")
            os.makedirs(input_dir, exist_ok=True)

    font = load_font(font_name, font_size)
    text_height = measure_text_height(font)
    line_height = text_height + line_spacing

    metadata_rows: list[dict[str, Any]] = []

    print(f"Rendering split '{split}' ({len(dataset)} samples) with image size {image_width}x{image_height}...")

    def get_render_text(prompt_text: str) -> str:
        """Get the text to render into the output image by stripping prompt_prefix."""
        if prompt_prefix and prompt_text.startswith(prompt_prefix):
            return prompt_text[len(prompt_prefix):]
        return prompt_text

    if num_workers > 0:
        # Pre-fetch text data for worker processes
        sample_args: list[tuple] = []
        pre_fetch_range = range(len(dataset))
        if tqdm is not None:
            pre_fetch_range = tqdm(pre_fetch_range, total=len(dataset), desc=f"[{split}] pre-fetch", unit="sample")
        for idx in pre_fetch_range:
            row = dataset[idx]
            prompt_text = str(row.get(prompt_column, ""))
            sample_args.append((
                idx, prompt_text, prompt_prefix, font_name, font_size,
                image_width, image_height, padding, line_spacing, paragraph_gap_lines,
            ))

        effective_cpus = _effective_cpu_count()
        max_workers = effective_cpus - 1
        if num_workers > max_workers:
            print(f"NOTE: --num_workers ({num_workers}) exceeds available cores "
                  f"(cgroup limit: {effective_cpus}, leaving 1 for main process). "
                  f"Auto-reducing to {max_workers}.")
            num_workers = max_workers

        mp_ctx = multiprocessing.get_context("spawn")
        worker_fn = render_sample_no_edit if no_edit_image else render_sample
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            result_iter = executor.map(worker_fn, sample_args, chunksize=100)
            if tqdm is not None:
                result_iter = tqdm(result_iter, total=len(dataset), desc=f"[{split}] rendering (×{num_workers})", unit="sample")
            for idx, input_png, output_png, img_size in result_iter:
                prompt_text = sample_args[idx][1]
                iw, ih = img_size
                if pack:
                    if shard_writer is None:
                        raise RuntimeError("Shard writer is not initialized")
                    shard_url = shard_writer.add_bytes(
                        input_png, output_png, iw, ih,
                        metadata={"prompt": prompt_text, "split": split, "index": idx},
                    )
                    image_ref = shard_url
                    edit_image_ref = "" if no_edit_image else shard_url
                else:
                    filename = f"{idx:06d}.png"
                    output_path = os.path.join(output_png_dir, filename)
                    with open(output_path, "wb") as f:
                        f.write(output_png)
                    image_ref = rel_path(output_path, split_dir)

                    if no_edit_image:
                        edit_image_ref = ""
                    else:
                        input_path = os.path.join(input_dir, filename)
                        with open(input_path, "wb") as f:
                            f.write(input_png)
                        edit_image_ref = rel_path(input_path, split_dir)

                if idx < 3:
                    from PIL import Image
                    output_image = Image.open(io.BytesIO(output_png))
                    output_image.save(os.path.join(examples_dir, f"{idx:06d}_output.png"))
                    if not no_edit_image:
                        input_image = Image.open(io.BytesIO(input_png))
                        input_image.save(os.path.join(examples_dir, f"{idx:06d}_input.png"))

                metadata_rows.append(
                    {"image": image_ref, "edit_image": edit_image_ref, "prompt": prompt_text,
                     "split": split, "index": idx}
                )
                if idx % 5000 == 4999:
                    mem = _cgroup_memory_gb()
                    if mem >= 0:
                        print(f"  [memory] {mem:.1f} GB at sample {idx+1}")
    else:
        iter_range = range(len(dataset))
        if tqdm is not None:
            iter_range = tqdm(iter_range, total=len(dataset), desc=f"[{split}] rendering", unit="sample")

        for idx in iter_range:
            row = dataset[idx]
            prompt_text = str(row.get(prompt_column, ""))
            render_text = get_render_text(prompt_text)

            if no_edit_image:
                input_image = None
            else:
                input_image = render_text_image(
                    prompt_text, image_width, image_height, font, line_height, padding, paragraph_gap_lines
                )
            output_image = render_text_image(
                render_text, image_width, image_height, font, line_height, padding, paragraph_gap_lines
            )
            _process_rendered_sample(
                idx, input_image, output_image, prompt_text,
                split, pack, shard_writer, split_dir, input_dir, output_png_dir, examples_dir,
                metadata_rows, no_edit_image=no_edit_image,
            )

    if pack and shard_writer is not None:
        shard_writer.finalize()

    metadata_path = os.path.join(split_dir, "metadata.csv")
    fieldnames = ["image", "edit_image", "prompt", "split", "index"]
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Split '{split}' saved to {split_dir}")
    print(f"  Samples: {len(metadata_rows)}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Example images: {examples_dir}/")