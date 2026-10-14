# TinyStories text rendering

Text rendering turns a story into an image-generation target.
This package reads TinyStories, wraps the text to a canvas, and writes
individual PNG images or packed PyTorch shards. The generator retains the
full prompt even when the rendered story is truncated by the canvas height.

The [long text rendering workflow](../../scripts/03_long_text_rendering/README.md)
provisions the frozen reproduction dataset from HuggingFace.
The commands below generate new samples from the repository root.
Matching the frozen release requires its rendering and selection settings;
regeneration alone does not establish byte equivalence.

## Text, images, and metadata

The generator prefixes each full story with
`Render these texts with a white background and black text: `.
That instruction and the story form the metadata `prompt`.
The target image renders the story without the instruction prefix.

By default, a condition image renders the complete prompt.
`--no_edit_image` omits that image and leaves `edit_image` empty.
This mode supplies text prompts to the [FLUX.2 text backend](../image_text/flux2/README.md).
The dedicated image-only loader requires a condition image and cannot
consume this mode as metadata pairs.

The CSV columns are `image`, `edit_image`, `prompt`, `split`, and `index`.
Image paths are relative to each output split directory.
Packed datasets use `shard:<start>:<index>` references and store targets
under `outputs` in each shard. An empty `edit_image` remains empty in packed
metadata, even when the shard contains an internal input placeholder.

Text is cleaned, wrapped, and truncated at complete lines that fit the
canvas. The default canvas is 512 by 512 pixels, with font size 13,
16-pixel padding, and four pixels of additional line spacing.
Font lookup can fall back to installed fonts, so an explicit font path
helps retain consistent glyphs across machines.

## Generate a small dataset

Install the repository dependencies and specify the rendering settings.
This example limits each requested source split to 100 samples.

```bash
pip install -r requirements.txt

python src/text_dataset/generate_tinystories.py \
  --dataset_name roneneldan/TinyStories \
  --splits train,validation --split_mapping train=train,validation=test \
  --max_samples 100 --no_edit_image \
  --image_width 384 --image_height 384 \
  --font_name src/long_multiplication_dataset/fonts/DejaVuSansMono.ttf \
  --font_size 13 --num_workers 0 \
  --output_dir data/tinystories_example
```

The default output mapping names HuggingFace `validation` as `test`.
`--shuffle` applies a shuffle with `--seed` before sample selection.
Without that flag, selection follows the loaded dataset order.
`--max_samples` limits each split independently.

Add `--pack --shard_size 10000` to store images in shards.
PNG output uses `images/output/` and, when enabled, `images/input/`.
Each split also has `metadata.csv` and example images.
Packed output uses `shards/` with a `shard_index.json` file.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--dataset_name` | `roneneldan/TinyStories` | HuggingFace text dataset |
| `--splits` | `train,validation` | Source splits |
| `--split_mapping` | `train=train,validation=test` | Output split names |
| `--max_samples` | Unlimited | Maximum samples per split |
| `--shuffle` | Disabled | Shuffle before selection |
| `--seed` | `0` | Shuffle seed |
| `--hf_endpoint` | `https://hf-mirror.com` | Dataset download endpoint |
| `--image_width`, `--image_height` | `512`, `512` | Canvas dimensions in pixels |
| `--font_name` | `Consolas` | Font name or file path |
| `--font_size` | `13` | Glyph size in pixels |
| `--padding` | `16` | Canvas margin |
| `--line_spacing` | `4` | Extra spacing after each line |
| `--paragraph_gap_lines` | `1` | Blank lines between paragraphs |
| `--no_edit_image` | Disabled | Use text conditioning without a condition image |
| `--pack` | Disabled | Write packed shards |
| `--shard_size` | `10000` | Samples per shard |
| `--num_workers` | `0` | Sequential rendering; a positive value uses worker processes |
| `--output_dir` | `output_tinystories` | Destination directory |

The canvas dimensions must be multiples of 16.
The text generator does not support `--image_height pad` because rendering
uses a fixed truncation height. The parallel renderer caps the requested
worker count according to the available CPU limit.

## Local scripts and source modules

The package-local scripts record additional generation settings.
`generate_300k_no_edit.sh` requests 300,000 samples per split on a 256-pixel
canvas. Its settings differ from the frozen 384-pixel reproduction release.
`example_generate_shard.sh` renders both source splits with packed storage.
Inspect those settings before using either script for a new dataset.

| Module or script | Responsibility |
| --- | --- |
| [generate_tinystories.py](generate_tinystories.py) | Dataset loading, split mapping, and command-line arguments |
| [convert_dataset.py](convert_dataset.py) | Rendering orchestration, metadata, and storage |
| [rendering.py](rendering.py) | Font loading, line wrapping, and canvas drawing |
| [_worker.py](_worker.py) | Rendering in worker processes |
| [shard_writer.py](shard_writer.py) | Packed PNG storage and shard index |
| [utils.py](utils.py) | Text cleanup and path helpers |
| [generate_300k_no_edit.sh](scripts/generate_300k_no_edit.sh) | Text-only generation settings |
| [example_generate_shard.sh](scripts/example_generate_shard.sh) | Packed generation settings |
