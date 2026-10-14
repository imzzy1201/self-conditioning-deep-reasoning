# Line tracing with alternating-group permutations

Line tracing represents a sequence of permutations as connected paths.
This generator samples permutations from the alternating group on five
symbols, A5. The condition image shows black wiring, and the target colors
each path according to its starting row.

The reproduction entry points are [line tracing](../../scripts/01_line_tracing/README.md)
and [inference-step scaling](../../scripts/04_nfe_scaling/line_tracing/README.md).
Their configurations define the experiment geometry and data sources.
The package-local scripts include generation checks and historical runs;
training scripts with cluster paths require adaptation for another checkout.

## Sample geometry

Each sample has five paths. Their starting rows use red, green, blue,
orange, and purple, respectively. The target preserves the condition wiring
and propagates the starting color along each complete trajectory.
The CSV stores `edit_image` for the condition, `image` for the target,
and an empty `prompt` for metadata compatibility.

The default patch size is 16 pixels, the canvas is 1024 by 80 pixels,
and permutation columns have stride four. Automatic length selection fits
15 permutation steps into 61 active columns. Nodes lie at row centers in
columns 1, 5, 9, and subsequent stride positions.

For each permutation, a segment connects a row in one column to its mapped
row in the next permutation column. The renderer draws crossing segments
in ascending order of signed row displacement. Segments with larger
downward displacement therefore appear above earlier segments.

Non-horizontal segments have width four pixels by default.
Horizontal segments and segments with absolute row displacement three
have width three pixels. The default node diameter is nine pixels.
Metadata records the geometry, permutation sequence, prefix compositions,
paths, palette, split, and sample index.

## Generate data

Run direct commands from the repository root.
This example creates 16 PNG samples with the default trace geometry.

```bash
python src/a5_line_trace_image_dataset/generate_dataset.py \
  --splits test --num_samples_per_split 16 \
  --num_permutations auto --image_width 1024 --column_stride 4 \
  --output_dir data/line_tracing_example --seed 0
```

Use `--pack --shard_size 10000` for packed storage.
The default split set is `train,val,test_random,test_depth`.
Training has 1,000,000 samples by default, and each other split has 1,000.
`--max_samples` overrides the count for every requested split.

An inclusive length range samples each non-depth-test instance uniformly.
The `test_depth` split instead cycles through `--depth_values`.
Use a canvas large enough for the largest requested length.

```bash
python src/a5_line_trace_image_dataset/generate_dataset.py \
  --splits train,val --num_train_samples 1000 --num_samples_per_split 100 \
  --num_permutations_range 4,15 --image_width 1024 --column_stride 4 \
  --output_dir data/line_tracing_variable --seed 0
```

Without `--adaptive_canvas`, shorter traces leave trailing white space.
With that flag, each canvas width fits the active columns exactly, and
`--image_width` acts as a maximum. Adaptive canvases produce variable
resolutions, which require batch size one during training.

A trace with 100 permutations and stride two needs 201 active columns.
The corresponding 16-pixel grid has width 3216 pixels.
The example uses a 3232-pixel canvas and explicit depth-test lengths.

```bash
python src/a5_line_trace_image_dataset/generate_dataset.py \
  --splits test_depth --num_samples_per_split 100 \
  --image_width 3232 --num_permutations 100 --column_stride 2 \
  --depth_values 20,40,60,80,100 \
  --output_dir data/line_tracing_long --seed 0
```

Generation writes a directory for each split.
PNG data and packed data share the same metadata schema.

```text
<split>/metadata.csv
<split>/examples/
<split>/images/input/     # PNG mode
<split>/images/output/    # PNG mode
<split>/shards/           # Packed mode, including shard_index.json
```

In packed mode, both image fields use the same `shard:<start>:<index>`
reference. [image_only/data.py](../image_only/data.py) reads the target
from `outputs` and the condition from `inputs`.

## Evaluate traces

The evaluator measures the longest correct prefix across all five paths.
A prefix of length `k` means that every path segment is correct for the
first `k` permutation steps. The summary field is
`max_correct_prefix_steps.mean`.

The evaluator also reports `step_accuracy.mean` and
`segment_accuracy.mean`. Step accuracy requires all five segments to be
correct at a step. Segment accuracy averages correctness over individual
path segments.

Generated ground-truth targets can check rendering and evaluation together.
The following command reads target references from the generated split.

```bash
python src/a5_line_trace_image_dataset/eval.py \
  --metadata_csv data/line_tracing_example/test/metadata.csv \
  --data_root data/line_tracing_example/test \
  --pred_root data/line_tracing_example/test \
  --output_json outputs/line_tracing_example/metrics.json \
  --output_jsonl outputs/line_tracing_example/per_sample.jsonl
```

For predictions, replace `--pred_root` with `--pred_dir /path/to/predictions`.
The evaluator accepts six-digit sample names and legacy five-digit prefixes.
It checks image size by default; `--no_strict_size` disables that check.

The local PNG and shard checks each generate 16 samples and score their
own targets. With the default 15-step geometry, expected means are 15 for
the longest prefix and 1.0 for both accuracy measures.

```bash
bash src/a5_line_trace_image_dataset/scripts/test_generate.sh
bash src/a5_line_trace_image_dataset/scripts/test_generate_shard.sh
```

## Package-local scripts and source

The full shard example generates 1,000,000 training samples and 10,000
samples per other split. The fixed-length scripts generate 500,000 packed
training samples and 100 PNG validation samples with length 30.
The validation script writes an inspection page for the paired images.

These scripts accept `OUT_DIR`; the training generator accepts `TRAIN_SEED`
and `SHARD_SIZE`, and the validation generator accepts `EVAL_SEED`.
Their names and explicit options record their generation settings.
Use the repository-level experiment workflows for current training runs.

| Module or script | Responsibility |
| --- | --- |
| [a5.py](a5.py) | Permutations and prefix composition |
| [renderer.py](renderer.py) | Black condition wiring and colored targets |
| [generate_dataset.py](generate_dataset.py) | Split sampling, metadata, and image output |
| [shard_writer.py](shard_writer.py) | Packed image storage |
| [eval.py](eval.py), [eval_plugin.py](eval_plugin.py) | Standalone scoring and training-time evaluation hook |
| [make_inspection_html.py](make_inspection_html.py) | Paired-image inspection page |
| [example_generate_shard.sh](scripts/example_generate_shard.sh) | Full generation settings |
| [generate_len_30_500k_train.sh](scripts/generate_len_30_500k_train.sh) | Fixed-length packed training set |
| [generate_len_30_eval_100.sh](scripts/generate_len_30_eval_100.sh) | Fixed-length validation set and inspection page |
| [scripts/](scripts/) | Local generation checks and historical training commands |
