# Long multiplication image datasets

A multiplication worksheet exposes intermediate arithmetic as a visual target.
This generator renders a problem image with two factors and a multiplication
sign. The paired target contains partial products, carries, successive sums,
and the final result.

The [long multiplication workflow](../../scripts/02_long_multiplication/README.md)
provisions the frozen reproduction data from HuggingFace.
The commands below generate new data from the repository root.
Regeneration does not establish byte equivalence with the frozen release.

## Arithmetic and rendering

Digits occupy square grid cells and align by place value.
The worksheet adds partial products successively, with two numbers in each
addition. Carry digits appear above and to the left of their associated cells.
Image dimensions are rounded up to multiples of 16.

The default font is the bundled [DejaVu Sans Mono](fonts/DejaVuSansMono.ttf).
An explicit `--font_name` can select another font name or path.
Rendering depends on the font and image library versions, so retain those
settings when comparing generated data. The layout specification is in
[DESIGN.md](DESIGN.md).

## Generate training and evaluation data

Install dependencies from the repository root.
The generator supports individual PNG images and packed PyTorch shards.

```bash
pip install -r requirements.txt

python src/long_multiplication_dataset/generate_random_digits_dataset.py \
  --min_digits 5 --max_digits 7 --num_samples 200000 \
  --output_dir data/long_multiplication/generated_train \
  --pack --shard_size 10000 --seed 42
```

Without arguments, the entry point generates 200,000 samples with factor
lengths between three and six digits under `output_random_digits/`.
Each factor length is sampled independently and uniformly.
`--balance_digits` instead distributes the sample count across length pairs.

The separate evaluation generator balances those pairs and writes PNG files.
Its defaults are 100 samples, five to seven digits, and seed 1201.
Pass training metadata to reject factor pairs that appear in that dataset.

```bash
python src/long_multiplication_dataset/generate_balanced_eval.py \
  --min_digits 5 --max_digits 7 --num_samples 100 \
  --exclude_metadata_csv data/long_multiplication/generated_train/metadata.csv \
  --output_dir data/long_multiplication/generated_eval \
  --seed 1201
```

With nine digit-length pairs and 100 samples, one pair receives 12 samples
and the other eight receive 11 each. Without `--exclude_metadata_csv`,
the generator does not check overlap with training factors.

The training generator's options control sampling and layout.
Run `--help` for all options.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--min_digits`, `--max_digits` | `3`, `6` | Inclusive factor-length bounds |
| `--base` | `10` | Arithmetic base |
| `--cell_size` | `16` | Grid cell size in pixels |
| `--image_width`, `--image_height` | Automatic | Explicit dimension or `pad` |
| `--font_name` | `DejaVu Sans Mono` | Font name or file path |
| `--padding` | `16` | Image margin in pixels |
| `--line_width` | Scaled from cell size | Horizontal line width |
| `--num_samples` | `200000` | Dataset size |
| `--balance_digits` | Disabled | Balance factor-length pairs |
| `--output_dir` | `output_random_digits` | Destination directory |
| `--seed` | Unset | Random seed |
| `--pack` | Disabled | Store images in packed shards |
| `--shard_size` | `10000` | Samples per packed shard |

An omitted dimension follows the sample content. A `pad` dimension uses
the canvas required by the maximum factor lengths. Explicit dimensions
must be multiples of 16. For example, add `--image_width 512
--image_height 512` to request a fixed canvas, or use `--base 16` for
hexadecimal arithmetic.

## Metadata and storage

The generated CSV has the columns `image`, `edit_image`, `prompt`, `a`, `b`,
`result`, `num_digits_a`, `num_digits_b`, and `base`.
The `image` field identifies the solution, `edit_image` identifies the
problem, and `prompt` is empty.

PNG output contains these paths relative to the dataset directory.

```text
metadata.csv
images/input/000000.png
images/output/000000.png
examples/000000_input.png
examples/000000_output.png
```

Packed output replaces the `images/` files with `shards/shard_000000.pt`
and a `shards/shard_index.json` index. Image fields contain references such
as `shard:000000:0`. The first example pair remains available as PNG files.

Use [image_only/data.py](../image_only/data.py) to load packed shards for
training. The package-local `ColumnMulDataset` reads PNG metadata or the
legacy single-file `data.pt` format; it does not resolve shard references.
For PNG data, it returns image tensors in `[0, 1]`.

```python
from src.long_multiplication_dataset.dataset import ColumnMulDataset

dataset = ColumnMulDataset("data/long_multiplication/generated_eval")
sample = dataset[0]
print(sample["a"], sample["b"], sample["result"])
print(sample["input"].shape, sample["output"].shape)
```

`ColumnMulDataset` accepts `transform_input` and `transform_output`.
The transforms run separately, so random transformations need explicit
coordination when paired images must stay aligned.

## Source modules

The implementation separates arithmetic, grid layout, rendering, and storage.
That separation permits direct inspection of a worksheet before generation.

| Module | Responsibility |
| --- | --- |
| [generate_random_digits_dataset.py](generate_random_digits_dataset.py) | Training-data command-line entry point |
| [generate_balanced_eval.py](generate_balanced_eval.py) | Balanced evaluation with optional overlap rejection |
| [config.py](config.py) | Configuration, dimension parsing, and default font |
| [arithmetic.py](arithmetic.py) | Partial products, sequential addition, and carries |
| [layout.py](layout.py) | Grid cells and horizontal rules |
| [renderer.py](renderer.py) | Problem and solution image rendering |
| [generator.py](generator.py) | Sample construction and generation utilities |
| [shard_writer.py](shard_writer.py) | Packed image storage |
| [dataset.py](dataset.py) | PNG and legacy tensor dataset loading |
