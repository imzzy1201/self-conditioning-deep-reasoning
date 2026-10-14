# Image-only flow matching for visual reasoning

Visual reasoning tasks can represent both a problem and its solution as images.
This package trains a FLUX.2-klein transformer to generate the solution image
from a condition image. The variational autoencoder (VAE) is frozen, and the
pipeline omits the text encoder and tokenizer.

The reproduction workflows for [line tracing](../../scripts/01_line_tracing/README.md)
and [long multiplication](../../scripts/02_long_multiplication/README.md) use
this package. Their configurations define the experiment settings. The commands
below describe direct use of the source modules from the repository root.

## Data and model inputs

The metadata loader requires `image` and `edit_image` columns. Include `prompt`
for compatibility with the generators; the model ignores its text.

```csv
image,edit_image,prompt
images/output/000000.png,images/input/000000.png,
```

The `image` field identifies the target, and `edit_image` identifies the
condition. Relative paths resolve against `--data_root`, or the CSV directory
when that argument is absent. Absolute image paths are accepted.

Packed datasets use `shard:000000:0` references in both image fields.
The loader reads `outputs` for the target and `inputs` for the condition.
It discovers packed data through `shards/shard_index.json` and supports PNG
bytes and the legacy unsigned-byte tensors.

Training center-crops the target to a multiple of `--size_divisor`, which
is 16 by default. The condition is resized and cropped to the target size.
Use `--batch_size 1` for variable resolutions. Larger batches and
`--compile` require uniform image dimensions.

The alternative `--dataset_type text` renders text into image pairs during
loading. For TinyStories, the first paragraph becomes the condition image,
and the full story becomes the target image. This mode still has no text
conditioning. Its default canvas is 512 by 512 pixels, which permits
larger batches when all samples use that size.

## Training

Install the repository dependencies before launching a CUDA run.

```bash
pip install -r requirements.txt

EXPERIMENT_METHOD=vanilla accelerate launch \
  --num_processes 1 --mixed_precision bf16 src/image_only/train.py \
  --metadata_csv /path/to/train/metadata.csv \
  --data_root /path/to/train \
  --output_dir outputs/manual_image_only \
  --batch_size 1 --num_epochs 5 --learning_rate 1e-4 \
  --mixed_precision bf16
```

Add `--use_lora --lora_rank 32 --lora_alpha 32` to train an adapter.
The default `--lora_target_modules` already lists the supported attention
and feed-forward projections. Set that argument only to change the target list.

Set `--from_scratch_config` to a preset or a transformer JSON configuration
for random initialization. Available presets are defined in
[configs.py](configs.py), including `9M_LONG`, `100M`, `300M`, `500M`, and `1B`.
The VAE remains frozen in this mode.

Text rendering can run directly from a HuggingFace dataset.
Use an explicit font path when rendering must use a particular font.

```bash
EXPERIMENT_METHOD=vanilla accelerate launch \
  --num_processes 1 --mixed_precision bf16 src/image_only/train.py \
  --dataset_type text --text_dataset_name roneneldan/TinyStories \
  --text_split train --text_max_samples 1000 \
  --text_image_width 512 --text_image_height 512 \
  --output_dir outputs/manual_rendered_text \
  --batch_size 4 --num_epochs 1 --mixed_precision bf16
```

The trainer records its arguments in `train_args.json`.
The principal defaults are listed below.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--dataset_type` | `metadata` | Read image metadata or render a text dataset |
| `--batch_size` | `1` | Number of samples per device |
| `--learning_rate` | `1e-4` | Optimizer learning rate |
| `--lr_scheduler` | `constant_with_warmup` | Learning-rate schedule |
| `--lr_warmup_steps` | `1000` | Warmup duration |
| `--gradient_accumulation_steps` | `1` | Batches per optimizer update |
| `--save_steps` | `5000` | Checkpoint interval in updates |
| `--log_every` | `100` | Console logging interval |
| `--num_workers` | `4` | Data loader workers |
| `--condition_time_offset` | `10` | Condition coordinate in image position IDs |
| `--preload_shards` | Disabled | Load all shards into memory |
| `--preload_shards_count` | `0` | Preload the first specified number of shards |
| `--dummy_text_embed_dim` | Inferred | Override the width of empty text embeddings |
| `--eval_every_steps` | `0` | Periodic evaluation interval; zero disables it |
| `--eval_num_visualize` | `5` | Samples visualized during periodic evaluation |

The scheduler accepts `linear`, `cosine`, `cosine_with_restarts`, `constant`,
and `constant_with_warmup`. Gradient checkpointing is enabled with
`--gradient_checkpointing`. It trades computation for lower memory use.
Weights & Biases logging uses `--use_wandb`, `--wandb_project`,
`--wandb_run_name`, and the optional `--wandb_entity`.

Periodic evaluation uses `--eval_metadata_csv`, `--eval_data_root`,
and `--eval_output_dir`. Set `--eval_every_steps` to a positive interval.
Use `--eval_disable_judge` when that evaluation does not need an API judge.

## Flow matching and self-conditioning

The `EXPERIMENT_METHOD` environment variable selects the loss and sampler
when [experiment_router.py](experiment_router.py) is imported.
The default method is `vanilla`; `proj_condition` enables self-conditioning.

Vanilla training interpolates between the target latent and Gaussian noise.
The transformer predicts the velocity, which is the noise minus the target.
Projection conditioning makes a first prediction, estimates the clean latent,
and uses that estimate in a second forward pass.

The FLUX.2 image sequence places target tokens first, condition tokens second,
and projection tokens last. Explicit image position IDs distinguish the
blocks. Text embeddings have shape `(B, 0, D)`, and `guidance_scale` is fixed
at `1.0`. Sampling runs through the method wrapper; the base pipeline's
`__call__` is disabled.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `AUX_LOSS` | `1` | Include the first-pass velocity loss |
| `AUX_LOSS_WEIGHT` | `0.5` | Weight of that auxiliary loss |
| `STOP_GRAD_PROJ` | `1` | Detach the projection before the second pass |
| `PROJ_TIME_OFFSET` | `-10` | Projection coordinate in image position IDs |

With auxiliary loss enabled, the loss is `(1 - w) * L_final + w * L_first`,
where `w` is `AUX_LOSS_WEIGHT`. Set `AUX_LOSS=0` to disable the first-pass term.
The image-only implementation drops the projection block when no estimate
is available. It does not read `ZERO_PROJ_WHEN_ABSENT`.

Use the same method and projection offset for training and inference.
FLUX.2 normalizes VAE latents with the frozen BatchNorm running statistics.
These statistics remain part of the model's encoding and decoding contract.

## Evaluation and checkpoints

Standalone evaluation accepts a base model, a saved transformer, or a LoRA
adapter. This example evaluates the transformer from the preceding training run.

```bash
EXPERIMENT_METHOD=vanilla python src/image_only/evaluation.py \
  --metadata_csv /path/to/test/metadata.csv \
  --data_root /path/to/test \
  --transformer_path outputs/manual_image_only/checkpoint-final/transformer \
  --output_dir outputs/manual_image_only_eval \
  --num_inference_steps 50 --num_visualize 8 --disable_judge
```

Use `--lora_path` for a LoRA adapter checkpoint.
Omit both checkpoint arguments to evaluate the base model.
Text-mode evaluation uses the same rendering arguments as training, with
`--text_split validation` for TinyStories.

The API judge reads `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL`
from `.env`. It uses `eval_prompt.txt` unless `--eval_prompt_path` specifies
another template. The default `--judge_k 2` requires agreement across calls.
Standalone evaluation accepts `--no-judge_strict` for majority voting.

Training writes `checkpoint-<step>/` at the save interval and
`checkpoint-final/` at completion. Each checkpoint contains `transformer/`
or `lora/` and `training_state.pt`. Accelerator state is enabled by default;
only the latest saved accelerator state is retained.

Resume with `--resume_from_checkpoint` pointing to a saved checkpoint.
The Boolean options `--save_accelerator_state` and `--save_final_checkpoint`
are enabled by default and have corresponding `--no-...` forms.

Evaluation writes generated images under `predictions/` and optional
frames and videos under `visualizations/`. API verdicts and token usage
are recorded in `judge_results.jsonl`. Dataset evaluation hooks, such as
the line-tracing metric, run through `eval_utils.py`.

## Source modules

The implementation separates data loading, losses, sampling, and evaluation.
The two design notes provide the corresponding derivation and data flow.

| Module | Responsibility |
| --- | --- |
| [_algorithm.md](_algorithm.md) | Flow-matching derivation and procedures |
| [_architecture.md](_architecture.md) | Components, dependencies, and data flow |
| [train.py](train.py) | Training, logging, checkpointing, and periodic evaluation |
| [evaluation.py](evaluation.py), [eval_utils.py](eval_utils.py) | Inference, visualization, and scoring |
| [data.py](data.py) | Metadata, packed shards, and online text rendering |
| [pipeline.py](pipeline.py), [rope_utils.py](rope_utils.py) | Image-only pipeline and position helpers |
| [vanilla_loss.py](vanilla_loss.py), [vanilla_sampling.py](vanilla_sampling.py) | Baseline loss and sampler |
| [proj_condition_loss.py](proj_condition_loss.py), [proj_condition_sampling.py](proj_condition_sampling.py) | Projection-conditioned loss and sampler |
| [judge.py](judge.py), [partial_product_judge.py](partial_product_judge.py) | API judging and arithmetic image scoring |
| [visualization.py](visualization.py), [bl_metrics.py](bl_metrics.py) | Denoising videos and optional field measurements |
