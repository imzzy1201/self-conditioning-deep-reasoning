# Text-conditioned flow matching with FLUX.2

Text and images can jointly specify a visual reasoning problem.
This backend trains FLUX.2-klein with a frozen Qwen3 text encoder and a frozen
VAE. The transformer predicts the solution image in latent space.
The default model is `black-forest-labs/FLUX.2-klein-base-4B`.

The [long text rendering](../../../scripts/03_long_text_rendering/README.md)
and [controlled DiffThinker](../../../scripts/05_diffthinker_controlled/README.md)
workflows use this backend. Direct commands below run from the repository
root. Set `PYTHONPATH=src/image_text:src` so the shared modules resolve.
The script directory supplies the backend-specific loss and sampler.

## Metadata and conditioning

The shared loader requires an `image` column for the target.
The optional `edit_image` column identifies a condition image, and `prompt`
contains text for Qwen3. Empty or absent condition fields are accepted.

```csv
image,edit_image,prompt
images/target_0001.png,images/condition_0001.png,Solve the worksheet.
images/target_0002.png,,Render the supplied story.
images/target_0003.png,images/condition_0003.png,
images/target_0004.png,,
```

Relative paths resolve against `--data_root`, or the metadata directory
when that argument is absent. Packed datasets use `shard:<start>:<index>`
references and a `shards/shard_index.json` file. The loader reads targets
from `outputs` and available condition images from `inputs`.

| Condition image | Prompt text | Model input |
| --- | --- | --- |
| Present | Present | Image and text conditioning |
| Present | Empty | Image conditioning with an empty prompt |
| Absent | Present | Text conditioning |
| Absent | Empty | No explicit problem content |

An empty prompt still passes through the text encoder's chat template.
Image-only inputs to this backend therefore use a different encoding path
from the dedicated [image-only package](../../image_only/README.md).

Targets retain their sample resolution after a center crop to the divisor,
which is 16 by default. Condition images are fitted to the target size.
Use `--batch_size 1` for variable resolutions. Larger batches and
`--compile` require uniform dimensions. The optional `--max_size` caps
the longest image side before divisor alignment.

## Training

Install the repository dependencies and launch the backend entry point.
This example trains a LoRA adapter with rank and alpha 32.

```bash
pip install -r requirements.txt

PYTHONPATH=src/image_text:src EXPERIMENT_METHOD=vanilla \
  accelerate launch --num_processes 1 --mixed_precision bf16 \
  src/image_text/flux2/train.py \
  --metadata_csv /path/to/train/metadata.csv \
  --data_root /path/to/train \
  --output_dir outputs/manual_flux2 \
  --batch_size 1 --num_epochs 5 --learning_rate 1e-4 \
  --use_lora --lora_rank 32 --lora_alpha 32 --mixed_precision bf16
```

Omit `--use_lora` to train the full transformer.
The text encoder follows `--transformer_model_name_or_path` unless
`--text_encoder_model_name_or_path` specifies another model.
Its default subfolder is `text_encoder`.

Random initialization uses `--from_scratch_config` with a preset or a
transformer JSON configuration. The available presets in
[configs.py](configs.py) include `100M`, `300M`, `500M`, and `1B`.
The VAE and text encoder remain frozen.

| Argument | Default | Purpose |
| --- | --- | --- |
| `--batch_size` | `1` | Samples per device |
| `--learning_rate` | `1e-4` | Optimizer learning rate |
| `--lr_scheduler` | `constant_with_warmup` | Learning-rate schedule |
| `--lr_warmup_steps` | `1000` | Warmup duration |
| `--gradient_accumulation_steps` | `1` | Batches per optimizer update |
| `--save_steps` | `5000` | Checkpoint interval |
| `--log_every` | `100` | Console logging interval |
| `--num_workers` | `4` | Data loader workers |
| `--condition_time_offset` | `10` | Condition coordinate in image position IDs |
| `--preload_shards` | Disabled | Load all packed shards into memory |
| `--preload_shards_count` | `0` | Preload the first specified number of shards |
| `--eval_every_steps` | `0` | Periodic evaluation interval |
| `--eval_num_visualize` | `8` | Samples visualized during periodic evaluation |

The scheduler accepts `linear`, `cosine`, `cosine_with_restarts`, `constant`,
and `constant_with_warmup`. Use `--gradient_checkpointing` to reduce
activation memory. Weights & Biases logging uses `--use_wandb`,
`--wandb_project`, `--wandb_run_name`, and the optional `--wandb_entity`.

Periodic evaluation uses `--eval_metadata_csv`, `--eval_data_root`,
and `--eval_output_dir`. Set `--eval_every_steps` to a positive interval.
The default periodic judge mode is `visual_full`; `--eval_judge_mode cer`
uses character error rate, and `--eval_judge_mode off` disables scoring.

## Self-conditioning and image positions

The shared [experiment router](../experiment_router.py) reads
`EXPERIMENT_METHOD` when imported. It selects `vanilla`, `proj_condition`,
or `iter_self_condition`, with `vanilla` as the default.

Vanilla flow matching predicts the velocity between target latents and
Gaussian noise. Projection conditioning estimates the clean target in a
first forward pass and conditions a second pass on that estimate.
Iterative self-conditioning repeats the projection procedure during training.
Its inference path uses the projection-conditioned sampler.

The image sequence contains target tokens, optional condition tokens,
and then projection tokens. FLUX.2 uses explicit image position IDs.
The condition offset defaults to `10`, and the projection offset to `-10`.
The VAE normalizes latents with its frozen BatchNorm running statistics.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `AUX_LOSS` | `1` | Include first-pass loss in `proj_condition` |
| `AUX_LOSS_WEIGHT` | `0.5` | Weight of the auxiliary term |
| `STOP_GRAD_PROJ` | `1` | Detach each projection before reuse |
| `ZERO_PROJ_WHEN_ABSENT` | `0` | Drop the unavailable projection block; `1` retains zeros |
| `PROJ_TIME_OFFSET` | `-10` | Projection coordinate in image position IDs |
| `NUM_ITERS` | `3` | Number of training passes in `iter_self_condition` |

With auxiliary loss enabled, `proj_condition` uses
`(1 - w) * L_final + w * L_first`, where `w` is `AUX_LOSS_WEIGHT`.
The iterative loss uses its own procedure in
[iter_self_condition_loss.py](iter_self_condition_loss.py).
Set the method, projection offset, and `ZERO_PROJ_WHEN_ABSENT` consistently
for training and inference.

The custom loss and sampling paths pass `guidance=None` to the transformer.
The documented runs use a guidance scale of `1.0`.
The command-line interface has no guidance-scale argument.

## Evaluation and saved state

Standalone evaluation can load a base model, a full transformer checkpoint,
or a LoRA adapter. This example evaluates the adapter from the training run.

```bash
PYTHONPATH=src/image_text:src EXPERIMENT_METHOD=vanilla \
  python src/image_text/flux2/evaluation.py \
  --metadata_csv /path/to/test/metadata.csv \
  --data_root /path/to/test \
  --lora_path outputs/manual_flux2/checkpoint-final/lora \
  --output_dir outputs/manual_flux2_eval \
  --num_inference_steps 50 --num_visualize 8 --judge_mode off
```

Use `--transformer_path` for full weights, or omit both checkpoint arguments
to evaluate the base model. `--experiment_method` can select the evaluation
method explicitly. It must agree with the training method.

The judge modes are `visual`, `visual_full`, `cer`, and `off`.
API modes read `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` from
`.env`. The default template is `eval_prompt_full.txt`, and
`--eval_prompt_path` overrides it. Judging defaults to two calls and
strict agreement.

Training writes `train_args.json`, periodic `checkpoint-<step>/` directories,
and `checkpoint-final/`. A checkpoint stores `transformer/` or `lora/`,
`training_state.pt`, and the latest retained `accelerator/` state.
Both final checkpoint saving and accelerator state saving are enabled
by default. Resume with `--resume_from_checkpoint`.

Evaluation writes `predictions/` and optional frames and videos under
`visualizations/`. Visual judging writes `judge_results.jsonl`, while
OCR judging writes `cer_results.jsonl`.
Checkpoint and step-count sweeps use
[evaluation_checkpoints.py](evaluation_checkpoints.py) and
[evaluation_steps_sweep.py](evaluation_steps_sweep.py).

## Source modules

Backbone-specific code stays in this directory. Data loading, judging,
evaluation control, and the training loop are shared under `src/image_text/`.

| Module | Responsibility |
| --- | --- |
| [_constants.py](_constants.py), [configs.py](configs.py) | Model defaults and transformer configurations |
| [train.py](train.py), [evaluation.py](evaluation.py) | Backend entry points and model construction |
| [vanilla_loss.py](vanilla_loss.py), [vanilla_sampling.py](vanilla_sampling.py) | Baseline loss and sampler |
| [proj_condition_loss.py](proj_condition_loss.py), [proj_condition_sampling.py](proj_condition_sampling.py) | Projection loss and sampler |
| [iter_self_condition_loss.py](iter_self_condition_loss.py) | Iterative training loss |
| [visualization.py](visualization.py) | Denoising visualization |
| [data.py](../data.py), [cli_args.py](../cli_args.py) | Shared metadata loader and arguments |
| [train_loop.py](../train_loop.py), [save_checkpoint.py](../save_checkpoint.py) | Shared optimization and checkpointing |
| [eval_utils.py](../eval_utils.py), [judge.py](../judge.py), [ocr_judge.py](../ocr_judge.py) | Shared inference and scoring |
