# Text-conditioned flow matching with QwenImage

QwenImage-Edit generates a solution image from a problem image and text.
This backend uses `Qwen/Qwen-Image-Edit-2511`, a frozen Qwen2.5-VL encoder,
and a frozen QwenImage VAE. The transformer supports full training and
LoRA adapters.

The [task-specific](../../../scripts/06_diffthinker_large_scale/task_specific/README.md)
and [unified](../../../scripts/06_diffthinker_large_scale/unified/README.md)
DiffThinker workflows use this backend. Data loading, judging, and training
control are shared with [FLUX.2](../flux2/README.md) under `src/image_text/`.
Run the commands below from the repository root with
`PYTHONPATH=src/image_text:src`.

## Data and latent representation

Metadata requires `image` for the target. The optional `edit_image` field
identifies the condition image, and `prompt` contains the text.
Relative paths resolve against `--data_root`, or the CSV directory when
that argument is absent. The shared loader accepts PNG files and packed
shards referenced by `shard:<start>:<index>`.

Qwen2.5-VL encodes the prompt together with the condition image.
For a training sample without `edit_image`, the encoder receives a gray
placeholder. This behavior requires task-specific validation before using
results from a task without condition images.

The VAE accepts `(B, C, 1, H, W)` tensors and normalizes each latent channel
with `vae.config.latents_mean` and `latents_std`. Packing combines each
two-by-two spatial patch into channels. Decoding selects the single
image frame from the temporal dimension.

Targets are aligned to a divisor of 16 by default. Condition images are
fitted to the target size. Keep `--batch_size 1` for variable resolutions;
larger batches and `--compile` require uniform image dimensions.

## Projection placement and positions

QwenImage derives rotary positions from the ordered `img_shapes` entries.
The image sequence is `[target, projection, condition]` when all three
blocks are present. The target stays first in every forward pass, and
velocity readout always uses `out[:, :target_len]`.

When no projection is available, `ZERO_PROJ_WHEN_ABSENT=0` removes that block.
Setting the variable to `1` retains a zero-filled projection block.
These choices change the sequence layout and must match between training
and inference. An absent condition image removes its latent block.

The shape list controls image frame coordinates. The transformer setting
`use_layer3d_rope` affects how the last entry receives a condition index.
Inspect the loaded transformer configuration when interpreting that layout.
`PROJ_TIME_OFFSET` and `--condition_time_offset` remain accepted for interface
consistency but have no numerical effect on this backend.

## Training and evaluation

Install the repository dependencies and launch a LoRA run.
The frozen VAE and text encoder supply latents and prompt embeddings.

```bash
pip install -r requirements.txt

PYTHONPATH=src/image_text:src EXPERIMENT_METHOD=proj_condition \
  AUX_LOSS=1 AUX_LOSS_WEIGHT=0.5 STOP_GRAD_PROJ=1 \
  ZERO_PROJ_WHEN_ABSENT=0 \
  accelerate launch --num_processes 1 --mixed_precision bf16 \
  src/image_text/qwenimage/train.py \
  --metadata_csv /path/to/train/metadata.csv \
  --data_root /path/to/train \
  --output_dir outputs/manual_qwenimage \
  --batch_size 1 --num_epochs 5 --learning_rate 1e-4 \
  --use_lora --lora_rank 32 --lora_alpha 32 --mixed_precision bf16
```

The method router accepts `vanilla`, `proj_condition`, and
`iter_self_condition`. Its default is `vanilla`.
The iterative method uses `NUM_ITERS=3` by default during training and
uses the projection-conditioned sampler for inference.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `AUX_LOSS` | `1` | Include the first-pass loss in `proj_condition` |
| `AUX_LOSS_WEIGHT` | `0.5` | Weight of the auxiliary term |
| `STOP_GRAD_PROJ` | `1` | Detach the projection before reuse |
| `ZERO_PROJ_WHEN_ABSENT` | `0` | Drop an unavailable projection block |
| `NUM_ITERS` | `3` | Training passes in `iter_self_condition` |

Evaluation loads a saved transformer with `--transformer_path` or an
adapter with `--lora_path`. Omit both arguments to use the base model.
Keep the training method and projection-layout setting for inference.

```bash
PYTHONPATH=src/image_text:src EXPERIMENT_METHOD=proj_condition \
  ZERO_PROJ_WHEN_ABSENT=0 \
  python src/image_text/qwenimage/evaluation.py \
  --metadata_csv /path/to/test/metadata.csv \
  --data_root /path/to/test \
  --lora_path outputs/manual_qwenimage/checkpoint-final/lora \
  --output_dir outputs/manual_qwenimage_eval \
  --num_inference_steps 50 --num_visualize 8 --judge_mode off
```

The shared API judge supports `visual`, `visual_full`, `cer`, and `off`.
API scoring reads `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL`
from `.env`. Structured DiffThinker scoring uses the separate evaluators
in `src/image_text/eval_external/` through the experiment workflows.

Training records `train_args.json` and writes periodic `checkpoint-<step>/`
directories and `checkpoint-final/`. Checkpoints contain `lora/` or
`transformer/`, `training_state.pt`, and the latest retained accelerator state.
Evaluation writes predictions and optional denoising visualizations.

## Source modules

Backend-specific encoding and sequence construction stay in this directory.
The shared modules are available through the required `PYTHONPATH` setting.

| Module | Responsibility |
| --- | --- |
| [train.py](train.py), [evaluation.py](evaluation.py) | Model loading and backend entry points |
| [_constants.py](_constants.py) | Model and subfolder defaults |
| [_latent_ops.py](_latent_ops.py) | Five-dimensional VAE encoding and normalization |
| [vanilla_loss.py](vanilla_loss.py), [vanilla_sampling.py](vanilla_sampling.py) | Baseline training and sampling |
| [proj_condition_loss.py](proj_condition_loss.py), [proj_condition_sampling.py](proj_condition_sampling.py) | Projection conditioning and target readout |
| [iter_self_condition_loss.py](iter_self_condition_loss.py) | Repeated projection during training |
| [evaluation_checkpoints.py](evaluation_checkpoints.py), [evaluation_steps_sweep.py](evaluation_steps_sweep.py) | Checkpoint and inference-step sweeps |
| [data.py](../data.py), [cli_args.py](../cli_args.py) | Shared data loading and arguments |
| [train_loop.py](../train_loop.py), [eval_utils.py](../eval_utils.py) | Shared training and evaluation control |
