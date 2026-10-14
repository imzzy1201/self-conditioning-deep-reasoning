# src/image_only/ Architecture Overview

## File Structure

```
src/image_only/
├── train.py                      # Training entry point
├── evaluation.py                 # Evaluation entry point (CLI)
├── eval_utils.py                 # Evaluation orchestrator
├── experiment_router.py          # [ROUTER] Selects loss/sampling method via env var
│
├── vanilla_loss.py               # Vanilla flow matching loss (baseline)
├── vanilla_sampling.py           # Vanilla denoising loop (baseline)
├── proj_condition_loss.py        # Projection-conditioned loss (experimental)
├── proj_condition_sampling.py    # Projection-conditioned sampling (experimental)
│
├── pipeline.py                   # Flux2KleinPipeline (structure + static utils, no __call__)
├── data.py                       # Dataset, collation, path resolution (PNG/Shard/Text)
├── configs.py                    # From-scratch transformer preset configs + JSON resolver
├── visualization.py              # Step-level video generation
├── judge.py                      # OpenAI vision API judge + parallel k-vote
├── eval_prompt.txt               # Prompt template for the judge (long_multiplication)
│
│   # GSM8k-specific judge prompt at src/gsm8k_aug_dataset/gsm8k_eval_prompt.txt
│   # Text rendering utility at src/text_dataset/rendering.py (used by data.py)
│
├── scripts/                      # Example and test shell scripts
├── _algorithm.md                 # Mathematical derivation of flow matching
├── _architecture.md              # This file
├── README.md                     # Usage documentation
```

Switch between methods via `EXPERIMENT_METHOD`:

```bash
EXPERIMENT_METHOD=vanilla        # baseline flow matching
EXPERIMENT_METHOD=proj_condition  # projection-conditioned (experimental)
```

## Dependency Graph

```
L0: data, judge, pipeline, configs                      ← leaf utilities
L1: vanilla_loss, vanilla_sampling
    proj_condition_loss, proj_condition_sampling         ← implementations
L2: experiment_router                                    ← import router
L3: eval_utils                                           ← eval orchestrator
L4: evaluation (CLI), train (CLI)                        ← entry points
```

No circular dependencies.

## Experiment Methods

`experiment_router.py` reads `EXPERIMENT_METHOD` and exports:
- `compute_loss` — the per-batch loss function
- `SamplingPipeline` — the denoising pipeline class

| Variable           | `vanilla`                                  | `proj_condition`                                        |
| ------------------ | ------------------------------------------ | ------------------------------------------------------- |
| `compute_loss`     | `vanilla_loss.compute_loss`                | `proj_condition_loss.compute_loss`                      |
| `SamplingPipeline` | `vanilla_sampling.VanillaSamplingPipeline` | `proj_condition_sampling.ProjConditionSamplingPipeline` |

Both methods share the same interface, so `train.py` and `eval_utils.py` switch transparently.

## Training

### Pipeline

```
metadata.csv / HF text dataset → MetadataImageEditDataset / TextImageEditDataset → DataLoader
                                              ↓
                              ┌────── compute_loss() ─────────────────────────────┐
                              │  1. VAE encode: image → 4D latents               │
                              │  2. Pack: 4D → [B, seq_len, C]                   │
                              │  3. Sample sigma ~ U(0, 1)                       │
                              │  4. x_t = (1-σ)·x_target + σ·noise               │
                              │  5. v_target = noise - x_target                  │
                              │  6. Build img_ids (target T=0, cond T=10)        │
                              │  7. Transformer forward(x_t, cond, t)            │
                              │  8. Slice velocity_pred[:seq_len]                │
                              │  9. MSE(velocity_pred, v_target)                 │
                              │ 10. backward() + clip_grad_norm() + step()       │
                              └──────────────────────────────────────────────────┘
                                              ↓
                              Every N steps: save checkpoint + optional eval
```

### Key Steps

1. **VAE Encoding**: Both target and condition images pass through the frozen VAE encoder. The VAE produces 4D latents `(B, C, H, W)` which are patchified (2×2 patches merged) and batch-norm normalized.
2. **Packing**: 4D latents `(B, C, H, W)` are reshaped to `[B, H*W, C]` — the transformer expects a flat sequence.
3. **Flow Matching**: `sigma ~ U(0, 1)` mixes noise into the target. Velocity target is `noise - target`.
4. **Conditioning via Position IDs**: Condition tokens have time channel set to `condition_time_offset` (default 10), target tokens use time=0.
5. **Prediction Slicing**: Only the first `seq_len` tokens (target portion) contribute to loss.

### Checkpoint Format

```
checkpoint-<step>/                  # Intermediate (saved every save_steps)
├── transformer/ (or lora/)         # Model weights via save_pretrained()
├── training_state.pt               # {"global_step": N, "epoch": E, "step_within_epoch": S}
└── accelerator/                    # Accelerator state for resume (optional, --save_accelerator_state)

checkpoint-final/                   # End-of-training (default, --save_final_checkpoint)
├── transformer/ (or lora/)         # Model weights
├── training_state.pt               # Same format
└── accelerator/                    # Accelerator state (optional)
```

**Accelerator state rotation**: When `--save_accelerator_state` is used, only the latest accelerator state is kept — old one is deleted after new save completes.
**Final checkpoint**: By default (`--save_final_checkpoint`, default True), a `checkpoint-final` directory is saved at training end. Disable with `--no-save_final_checkpoint` (useful when the last step-based checkpoint needs to retain accelerator state for resume).

### Training Features

- **W&B integration** (`--use_wandb`): Logs loss, accuracy, time/total_sec (wall time), time/train_sec (pure training time excluding eval/judge), and periodic eval metrics.
- **Periodic evaluation** (`--eval_every_steps`): Runs inference + judge on validation set during training, reports accuracy to wandb. Eval time subtracted from `time/train_sec`.
- **Token usage tracking**: Judge API token consumption accumulated across all periodic evals and final evaluation.
- **From-scratch training** (`--from_scratch_config`): Creates a random-init transformer from preset configs (100M/300M/500M/1B) or a JSON file. Mutually exclusive with `--use_lora`.
- **Gradient checkpointing** (`--gradient_checkpointing`): Saves memory at ~20-30% speed cost.
- **torch.compile** (`--compile`): Compiles transformer after LoRA + gradient checkpointing, before accelerator.prepare. Requires fixed-resolution data.
- **Checkpoint resume** (`--resume_from_checkpoint`): Restores model weights, optimizer, scheduler, and RNG state.
- **Text dataset** (`--dataset_type text`): On-the-fly rendering from HuggingFace text datasets via `TextImageEditDataset`. Fixed resolution enables `batch_size > 1`.

## Evaluation

`evaluate_rows_with_pipeline()` in `eval_utils.py` auto-wraps the base `Flux2KleinPipeline` in `SamplingPipeline`. Accepts both `MetadataImageEditDataset` and `TextImageEditDataset`.

### Pipeline

```
metadata.csv / HF text dataset → MetadataImageEditDataset / TextImageEditDataset → for each sample:
    1. dataset.get_pil_sample(idx) → load condition + answer images
    2. Create generator with per-sample seed
    3. SamplingPipeline(image=condition, num_inference_steps=N) → pred_image
    4. Save pred_image to predictions/
    5. If visualization: collect step logs → build_visualization_artifacts()
                                               ↓
                              After loop: run_judge_in_parallel(samples, k=N, strict=True)
                                               ↓
                              Summarize results → write judge_results.jsonl
```

### Judge Voting

- `k` calls per sample, all parallelized in one ThreadPoolExecutor pool
- `strict=True` (default): all k votes must agree, otherwise error
- `strict=False`: majority vote, ties default to "no"
- JSONL includes `judge_details` array when k > 1
- Progress bar shown during parallel judging
- Token usage (`prompt_tokens`, `completion_tokens`) tracked per vote and summed in summary

### Key Components

- **`get_generation_size()`**: Determines output dimensions from the condition image.
- **`evaluate_rows_with_pipeline()`**: Core loop — iterates rows, runs inference, optionally visualizes and judges.
- **Step callback**: For the first `num_visualize` samples, captures `(latents_pre_step, noise_pred, timestep)` at each step.

## Visualization

### Three Video Types

1. **`denoising.mp4`**: Decodes each step's raw latent directly to pixel space.
2. **`linear_to_final.mp4`**: Projects each step to the final image using `x_final = x_t - σ · noise_pred`.
3. **`interp_noise_to_answer_linear_to_final.mp4`**: Interpolates noise→answer, projects to final image at each point.

## Judge

- Reads `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` from environment/.env
- Sends both images as base64 data URLs to an OpenAI-compatible vision API
- Parses `[YES]`/`[NO]` from response with 3-retry logic
- Stores `reasoning_content` when the API returns it
- Prompt template customizable via `--eval_prompt_path` (default: `eval_prompt.txt`; GSM8k data uses `src/gsm8k_aug_dataset/gsm8k_eval_prompt.txt` which compares only final answers, not intermediate steps)

## Data

- **`MetadataImageEditDataset`**: Loads `(image, edit_image)` pairs from `metadata.csv`. Supports two modes:
  - **PNG mode** (default): Reads individual image files via PIL. Path resolved from `image`/`edit_image` columns relative to `data_root`.
  - **Shard mode** (auto-detected): When `shards/shard_index.json` exists in `data_root`, reads from shard `.pt` files via `ShardCache` (lazy-loading, max_cached=2 shards, FIFO eviction). Shard URL format: `shard:NNNNNN:idx` in `image`/`edit_image` columns — same URL for both, Dataset reads from `inputs`/`outputs` key based on column. Supports both **PNG-bytes format** (`list[bytes]` per shard, ~20x smaller disk) and **uint8-tensor format** (legacy `Tensor[N,3,H,W]`). Auto-detected per entry via `isinstance(entry, bytes)`.
  - **Shard preload**: `--preload_shards` loads all shards into memory at init (eliminates per-item torch.load overhead); `--preload_shards_count N` loads only N shards.
- **`TextImageEditDataset`**: On-the-fly rendering from HuggingFace text datasets. Each sample's prompt/response text is rendered into images at `__getitem__` time, using `src/text_dataset/rendering.py`. Fixed resolution (`--text_image_width`, `--text_image_height`), enabling `batch_size > 1`. Supports custom extract functions and direct prompt/response columns.
- **`get_pil_sample(idx)`**: Returns `{"image": PIL, "edit_image": PIL, "metadata": dict}` — unified PIL access used by eval_utils.py. Both dataset classes implement this interface.
- **`collate_image_edit_batch`**: Stacks samples. Enforces uniform dimensions within a batch (variable-resolution data requires `batch_size=1`).
- **`load_rgb_image`**: Simple PIL RGB image loader for file paths (used by visualization.py).

## Pipeline

Modified from `diffusers 0.37.1: Flux2KleinPipeline`. Key changes:
- Text encoder forced to `None` (no Qwen3 loaded)
- `encode_prompt` returns empty embeddings
- `_get_dummy_text_embed_dim`: infers empty text embedding dimension from `transformer.config.joint_attention_dim` or `context_embedder.in_features`. Override via `--dummy_text_embed_dim`.
- `guidance_scale` defaults to 1.0, CFG disabled
- `__call__` removed — use `SamplingPipeline` instead
- `Flux2KleinPipeline` only provides structure and static utility methods

## Vanilla (Baseline)

Vanilla flow matching: single forward pass `v_theta(x_t, t | {z})` with MSE loss.

- **Loss**: `vanilla_loss.py` — one transformer forward, MSE(v_pred, v_target)
- **Sampling**: `vanilla_sampling.py` — standard Euler scheduler loop, no projection condition

## Projection-Conditioned (Experimental)

Each denoising step uses the previous step's projected result as an extra condition to reuse already-inferred information.

- **Loss**: `proj_condition_loss.py` — two forward passes: first get v_z, project to x_{t#}, then forward with z + x_{t#}
- **Sampling**: `proj_condition_sampling.py` — cache x_{s#} from previous step as third condition token

## From-Scratch Configs

`configs.py` defines `FROM_SCRATCH_CONFIGS` — preset architecture dicts for `Flux2Transformer2DModel.from_config()` (random init, no pretrained weights). Also supports custom JSON files via `resolve_from_scratch_config()`.

| Preset | inner_dim | heads | double | single | joint_dim | guidance | Params  |
|--------|-----------|-------|--------|--------|-----------|----------|---------|
| 100M   | 512       | 4     | 4      | 20     | 512       | False    | 100.7M  |
| 300M   | 1024      | 8     | 4      | 14     | 1024      | False    | 320.3M  |
| 500M   | 1280      | 10    | 4      | 14     | 1280      | False    | 500.4M  |
| 1B     | 1536      | 12    | 4      | 22     | 1536      | False    | 965.7M  |

`joint_attention_dim = inner_dim` because no-text mode passes 0 context tokens; context_embedder and cross-attention in double-stream blocks have zero gradients (~14% dead weight per double block). `guidance_embeds=False` because `guidance_scale=1.0`.

Mutually exclusive with `--use_lora`. Checkpoint save/resume works via `save_pretrained` → `load_state_dict`.

### Debug Visualization

Set `__PROJ_COND_DEBUG__=1` to save VAE-decoded projection images:
- **Training**: `temp/train_proj/step_XXXX.png` — circular buffer of 10 slots, auto-initialized from `vae_encoder`
- **Sampling**: `temp/sampling_proj/step_XXXX.png` — full trajectory of $x_{\sigma\#}$ per step
