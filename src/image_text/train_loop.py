"""Shared training driver (backbone-agnostic).

`run_train_main` owns the entire training scaffold — accelerator/dtype/wandb
setup, LoRA / gradient-checkpointing / torch.compile, optimizer, dataloader,
lr_scheduler, `accelerator.prepare`, resume, the epoch/batch loop, checkpointing,
and periodic eval. Backbone-specific steps are injected as callbacks:

  load_models(args, model_dtype, accelerator) -> models
      Load transformer/VAE/text-encoder/tokenizer[/processor] + build the
      `vae_encoder` shim. Returns a namespace exposing `.transformer` (raw,
      pre-LoRA) and `.vae_encoder` plus whatever build_eval_pipe / prepare_batch
      need (frozen `.vae`, `.text_encoder`, `.tokenizer`, `.processor`).

  prepare_batch(batch, models, args, model_dtype, device)
      Encode the (vision-grounded) prompt embeddings into `batch` (e.g.
      `batch["prompt_embeds"]` / `batch["prompt_embeds_mask"]` / `batch["text_ids"]`).

  build_eval_pipe(eval_transformer, models, model_dtype, device) -> pipe
      Construct the periodic-eval pipeline from the live (unwrapped) transformer
      and the frozen model stack.

`compute_loss` / `extra_args` are passed in by the caller (resolved from
`experiment_router` so the selected backend/method is used).
"""

import json
import math
import os
import time
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from tqdm.auto import tqdm

from data import (
    MetadataImageEditDataset,
    collate_image_edit_batch,
)
from experiment_tools import flops as flops_tools
from save_checkpoint import save_checkpoint
from periodic_eval import run_periodic_eval


def run_train_main(args, *, extra_args, compute_loss, load_models, prepare_batch, build_eval_pipe):
    # Inject experiment-specific extra args into args.experimental.
    setattr(args, "experimental", extra_args)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.log_every < 0:
        raise ValueError("--log_every must be >= 0")
    if args.save_steps < 0:
        raise ValueError("--save_steps must be >= 0")
    if args.eval_every_steps < 0:
        raise ValueError("--eval_every_steps must be >= 0")
    if args.metadata_csv is None:
        raise ValueError("--metadata_csv is required")
    if args.eval_every_steps > 0 and args.eval_metadata_csv is None:
        raise ValueError("--eval_metadata_csv is required when --eval_every_steps > 0")
    if args.eval_every_steps > 0 and args.eval_num_inference_steps <= 0:
        raise ValueError("--eval_num_inference_steps must be > 0 when --eval_every_steps > 0")
    if args.from_scratch_config is not None and args.use_lora:
        raise ValueError("--from_scratch_config and --use_lora are mutually exclusive. "
                         "From-scratch training trains all params; LoRA freezes the base and trains adapters.")
    if args.from_scratch_config is not None:
        args.from_scratch_config_dict = resolve_from_scratch_config_local(args.from_scratch_config)
    else:
        args.from_scratch_config_dict = None
    if args.preload_shards and args.preload_shards_count > 0:
        raise ValueError("--preload_shards and --preload_shards_count > 0 are mutually exclusive. "
                         "Use --preload_shards to load all shards, or --preload_shards_count N to load only N.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
    # Raise the collective timeout well above NCCL's 600s default. Periodic eval
    # has unavoidable rank-time asymmetry: the centralized judge runs only on the
    # main process (300 samples x k API calls) while other ranks block at the
    # post-eval wait_for_everyone(). Under the default timeout that barrier can
    # trip the watchdog and tear down training. Env-overridable.
    nccl_timeout_sec = int(os.environ.get("NCCL_TIMEOUT_SEC", "3600"))
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout_sec))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
    )
    accelerator.print("[init] Accelerator initialized")
    accelerator.print(
        f"[init] device={accelerator.device}, mixed_precision={accelerator.mixed_precision}, "
        f"grad_accum={args.gradient_accumulation_steps}"
    )

    if args.seed is not None:
        set_seed(args.seed)
        accelerator.print(f"[init] Seed set to {args.seed}")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        args_path = os.path.join(args.output_dir, "train_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
        accelerator.print(f"[init] Saved run config to: {args_path}")

    if args.use_wandb:
        wandb_init_kwargs = {}
        if args.wandb_run_name is not None:
            wandb_init_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity is not None:
            wandb_init_kwargs["entity"] = args.wandb_entity

        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": wandb_init_kwargs} if len(wandb_init_kwargs) > 0 else None,
        )
        accelerator.print(
            f"[init] W&B enabled: project={args.wandb_project}, run_name={args.wandb_run_name}, "
            f"entity={args.wandb_entity}"
        )

    if accelerator.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32
    if accelerator.device.type != "cuda":
        model_dtype = torch.float32
    accelerator.print(f"[init] Runtime model dtype: {model_dtype}")

    # Backbone-specific loading (transformer / VAE / text-encoder / vae_encoder shim).
    models = load_models(args, model_dtype, accelerator)
    transformer = models.transformer
    vae_encoder = models.vae_encoder

    if args.use_lora:
        accelerator.print("[init] LoRA enabled, preparing PEFT adapters...")
        transformer.requires_grad_(False)
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        if len(target_modules) == 0:
            raise ValueError("--lora_target_modules must contain at least one module name")
        accelerator.print(f"[init] LoRA target modules: {len(target_modules)}")

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        accelerator.print("[init] LoRA adapters attached")
    elif args.from_scratch_config_dict is not None:
        accelerator.print("[init] From-scratch training: all transformer params trainable")

    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        accelerator.print("[init] Gradient checkpointing enabled")
        # LoRA (and from-scratch-with-frozen-base) freezes the backbone, so the
        # inputs to each checkpointed block have requires_grad=False. Without
        # forcing an input to require grad, the checkpoint boundary won't
        # propagate grads to the trainable adapter params INSIDE each block ->
        # LoRA silently stops learning. This is the standard PEFT + grad-ckpt fix.
        if hasattr(transformer, "enable_input_require_grads"):
            transformer.enable_input_require_grads()
            accelerator.print("[init] enable_input_require_grads set (grad-ckpt + trainable adapters)")
    transformer.train()

    # torch.compile: apply after LoRA + gradient checkpointing, before accelerator.prepare()
    if args.compile:
        accelerator.print("[init] Applying torch.compile to transformer")
        transformer = torch.compile(transformer)
        accelerator.print("[init] torch.compile applied")

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check LoRA settings.")
    trainable_numel = sum(p.numel() for p in trainable_params)
    total_numel = sum(p.numel() for p in transformer.parameters())
    trainable_ratio = 100.0 * trainable_numel / max(total_numel, 1)
    accelerator.print(
        f"[init] Trainable params: {trainable_numel:,} / {total_numel:,} ({trainable_ratio:.4f}%)"
    )

    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
    )
    accelerator.print(f"[init] Optimizer AdamW ready, lr={args.learning_rate:.2e}")

    dataset = MetadataImageEditDataset(
        metadata_csv=args.metadata_csv,
        data_root=args.data_root,
        size_divisor=args.size_divisor,
        preload_shards=args.preload_shards,
        preload_shards_count=args.preload_shards_count,
        target_mode=args.target_mode,
        target_height=args.target_height,
        target_width=args.target_width,
        depth_normalization=args.depth_normalization,
        max_size=args.max_size,
    )
    # Disable shuffle when using partial shard preload to avoid thrashing the LRU cache.
    partial_preload = (
        dataset._shard_mode
        and not args.preload_shards
        and args.preload_shards_count > 0
        and args.preload_shards_count < dataset._shard_cache.total_shards
    )
    should_shuffle = not partial_preload

    # Shard mode + num_workers > 0: each DataLoader worker fork gets an independent
    # ShardCache. With large shards (~2.6GB), this multiplies RAM usage by num_workers
    # and causes OOM on multi-GPU. Recommend num_workers=0 for shard mode.
    if dataset._shard_mode and args.num_workers > 0 and accelerator.num_processes > 1:
        accelerator.print(
            f"[WARN] Shard mode + num_workers={args.num_workers} + {accelerator.num_processes} GPUs: "
            f"each worker fork inherits an independent ShardCache, multiplying RAM usage. "
            f"Recommend --num_workers 0 for shard mode to avoid CPU OOM."
        )

    accelerator.print(
        f"[init] Dataset: {len(dataset)} samples, "
        f"shuffle={should_shuffle}"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=should_shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_image_edit_batch,
    )
    accelerator.print(
        f"[init] Dataset samples={len(dataset)}, batch_size={args.batch_size}, num_workers={args.num_workers}"
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / accelerator.num_processes / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    accelerator.print(
        f"[init] updates_per_epoch={num_update_steps_per_epoch}, num_epochs={args.num_epochs}, "
        f"max_train_steps={args.max_train_steps}"
    )

    # accelerator.prepare() wraps the scheduler so that each .step() advances the
    # underlying scheduler num_processes times (split_batches=False default). To make
    # warmup/decay span the intended number of *optimizer* steps, scale the step
    # counts by num_processes here. Without this, an N-GPU run consumes the schedule
    # N× too fast (e.g. linear decay hits LR=0 at max_train_steps/N).
    lr_scheduler = get_scheduler_local(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    accelerator.print(
        f"[init] Scheduler ready: type={args.lr_scheduler}, warmup={args.lr_warmup_steps} "
        f"(scaled x{accelerator.num_processes} procs internally)"
    )

    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )
    accelerator.print("[init] Accelerator.prepare complete")

    # --- Resume from checkpoint ---
    resumed_epoch = 0
    resumed_step_within_epoch = 0
    if args.resume_from_checkpoint is not None:
        resume_dir = args.resume_from_checkpoint
        if not os.path.isdir(resume_dir):
            raise ValueError(f"--resume_from_checkpoint directory does not exist: {resume_dir}")

        state_path = os.path.join(resume_dir, "training_state.pt")
        if not os.path.isfile(state_path):
            raise ValueError(f"training_state.pt not found in checkpoint dir: {resume_dir}")

        training_state = torch.load(state_path, map_location="cpu", weights_only=True)
        global_step = training_state["global_step"]
        resumed_epoch = training_state.get("epoch", 0)
        resumed_step_within_epoch = training_state.get("step_within_epoch", 0)
        accelerator.print(
            f"[resume] Loaded training_state.pt: global_step={global_step}, "
            f"epoch={resumed_epoch}, step_within_epoch={resumed_step_within_epoch}"
        )

        # Restore model weights from the checkpoint subdirectory.
        model_subdir = "lora" if args.use_lora else "transformer"
        model_path = os.path.join(resume_dir, model_subdir)
        if os.path.isdir(model_path):
            accelerator.print(f"[resume] Loading model weights from: {model_path}")
            unwrapped = accelerator.unwrap_model(transformer)
            # Strip torch.compile wrapper so load_state_dict keys match checkpoint.
            if args.compile and hasattr(unwrapped, '_orig_mod'):
                unwrapped = unwrapped._orig_mod
            if args.use_lora:
                unwrapped.load_adapter(model_path, adapter_name="default")
            else:
                # Load tensors from safetensors to CPU, then copy in-place into
                # existing GPU parameters — avoids allocating a second copy on GPU.
                safetensors_path = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
                loaded_state_dict = load_file(safetensors_path)
                loaded_state_dict = {k: v.to(dtype=model_dtype) for k, v in loaded_state_dict.items()}
                unwrapped.load_state_dict(loaded_state_dict)
        else:
            accelerator.print(
                f"[resume] WARNING: model subdir not found at {model_path}, "
                "skipping model weight restore"
            )

        # Restore optimizer / scheduler / RNG via Accelerator.
        accel_state_path = os.path.join(resume_dir, "accelerator")
        if os.path.isdir(accel_state_path):
            # map_location="cpu": load the 22B model state to CPU first, then
            # copy into the GPU params. For multi-GPU, accelerate defaults to
            # map_location="on_device", so safetensors load_file allocates the
            # full ~41GB model ON the GPU — a second copy alongside the already-
            # resident model -> OOM on 8xA100-80G during resume (fresh training
            # never calls load_state, so it never hit this). CPU-load + per-param
            # copy is transient and fits. Optimizer/scheduler/RNG unaffected.
            accelerator.load_state(accel_state_path, map_location="cpu")
            accelerator.print(f"[resume] Restored optimizer/scheduler/RNG from: {accel_state_path}")
        else:
            accelerator.print(
                "[resume] WARNING: accelerator state dir not found, "
                "optimizer/scheduler/RNG NOT restored"
            )

        accelerator.print(f"[resume] Resuming from global_step={global_step}")
    else:
        global_step = 0

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="train")
    progress_bar.update(global_step)
    total_start = time.perf_counter()
    eval_sec_total = 0.0
    last_accel_path: str | None = None

    # --- FLOP measurement state ---
    # cumulative_flops is a cluster-total (per-GPU per-micro-batch x
    # num_processes), so FLOP-matched pairing reflects total compute consumed,
    # not one replica's.
    _flops_measured: list[float] = []
    _flops_per_micro_batch: float | None = None
    _flops_cumulative: float = 0.0
    _ngpus = accelerator.num_processes
    if args.measure_flops and args.resume_from_checkpoint is not None:
        # stored value is already cluster-total; increments below are too, so
        # resuming stays consistent.
        _flops_cumulative = float(training_state.get("cumulative_flops", 0.0))

    # External measured rate for wandb FLOP logging (real run trains at full
    # speed without a counter; use the poke-measured rate to log the running
    # cluster-total cumulative each step).
    _flops_log_rate: float | None = None
    if args.flops_rate_source is not None:
        _fj = flops_tools.load_flops_json(args.flops_rate_source)
        if not _fj or "per_micro_batch_flops" not in _fj:
            raise ValueError(
                f"--flops_rate_source {args.flops_rate_source} has no per_micro_batch_flops; "
                "run the 100-micro-batch --measure_flops poke first."
            )
        _flops_log_rate = float(_fj["per_micro_batch_flops"])
        accelerator.print(
            f"[flops] logging cumulative to wandb from measured rate={_flops_log_rate:.3e}"
        )
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if args.eval_every_steps > 0:
        accelerator.print(
            f"[eval] Periodic eval enabled: every {args.eval_every_steps} steps, "
            f"metadata={args.eval_metadata_csv}, max_samples={args.eval_max_samples}, "
            f"judge={'off' if args.eval_disable_judge else 'on'}"
        )

    # Evaluate before training if requested
    if args.eval_before_training and args.eval_every_steps > 0 and global_step == 0:
        accelerator.print("[eval] Running evaluation before training (step 0)...")
        eval_metrics = run_periodic_eval(
            accelerator, args, transformer, models, build_eval_pipe, global_step, model_dtype
        )
        if accelerator.is_main_process and eval_metrics:
            for k, v in eval_metrics.items():
                accelerator.log({k: v}, step=global_step)
            if args.use_wandb:
                import wandb
                wandb.log(eval_metrics, step=global_step)
            # Track judge token usage
            total_prompt_tokens += eval_metrics.get("eval/total_prompt_tokens", 0)
            total_completion_tokens += eval_metrics.get("eval/total_completion_tokens", 0)

    for epoch in range(args.num_epochs):
        accelerator.print(f"[train] Epoch {epoch + 1}/{args.num_epochs} started")
        step_within_epoch = 0

        # --- Explicit epoch skipping ---
        # FIXME: epoch skipping is slow and arguably wrong — accelerator.load_state already restores RNG, so this doesn't reproduce original shuffle order. Consider starting from resumed_epoch directly and accepting minor data repetition.
        if epoch < resumed_epoch:
            for _ in dataloader:
                pass
            accelerator.print(f"[resume] Skipped epoch {epoch + 1} (already completed)")
            continue

        # FIXME: step-level skipping is slow (one-by-one consumption). Consider removing and accepting minor data repetition in the resumed epoch.
        if epoch == resumed_epoch and resumed_step_within_epoch > 0:
            batches_to_skip = resumed_step_within_epoch * args.gradient_accumulation_steps
            accelerator.print(
                f"[resume] Skipping {batches_to_skip} micro-batches "
                f"({resumed_step_within_epoch} optimizer steps) in epoch {epoch + 1}"
            )
            data_iter = iter(dataloader)
            for _ in range(batches_to_skip):
                try:
                    next(data_iter)
                except StopIteration:
                    break
            step_within_epoch = resumed_step_within_epoch
            resumed_step_within_epoch = 0
        else:
            data_iter = iter(dataloader)

        for batch in data_iter:
            with accelerator.accumulate(transformer):
                # Backbone-specific prompt encoding (vision-grounded for VLM backends).
                prepare_batch(batch, models, args, model_dtype, accelerator.device)

                measuring = args.measure_flops and _flops_per_micro_batch is None
                if measuring:
                    with FlopCounterMode() as _fc:
                        loss = compute_loss(
                            accelerator=accelerator,
                            transformer=transformer,
                            batch=batch,
                            vae_encoder=vae_encoder,
                            model_dtype=model_dtype,
                            condition_time_offset=args.condition_time_offset,
                        )
                        accelerator.backward(loss)
                    _flops_measured.append(_fc.get_total_flops())
                    _flops_cumulative += _flops_measured[-1] * _ngpus
                    if len(_flops_measured) >= flops_tools.MEASUREMENT_MICRO_BATCHES:
                        _flops_per_micro_batch = sum(_flops_measured) / len(_flops_measured)
                        flops_tools.write_flops_json(
                            args.output_dir,
                            task=args.flops_task, method=extra_args.get("method", ""),
                            model_size=args.flops_model_size, seed=getattr(args, "seed", 0),
                            per_micro_batch_flops=_flops_per_micro_batch,
                        )
                        accelerator.print(
                            f"[flops] per_micro_batch_flops={_flops_per_micro_batch:.3e} "
                            f"(from {len(_flops_measured)} micro-batches) "
                            f"cumulative={_flops_cumulative:.3e}"
                        )
                else:
                    loss = compute_loss(
                        accelerator=accelerator,
                        transformer=transformer,
                        batch=batch,
                        vae_encoder=vae_encoder,
                        model_dtype=model_dtype,
                        condition_time_offset=args.condition_time_offset,
                    )
                    accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                step_within_epoch += 1
                progress_bar.update(1)

                if args.log_every > 0 and global_step % args.log_every == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    total_sec = time.perf_counter() - total_start
                    train_sec = total_sec - eval_sec_total
                    samples_seen = global_step * args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
                    accelerator.print(f"step={global_step} loss={loss.item():.6f} lr={lr:.6e} samples={samples_seen}")
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/samples_seen": samples_seen,
                        "time/total_sec": total_sec,
                        "time/train_sec": train_sec,
                    }
                    if _flops_log_rate is not None:
                        log_dict["train/cumulative_flops"] = (
                            _flops_log_rate * global_step
                            * args.gradient_accumulation_steps * accelerator.num_processes
                        )
                else:
                    log_dict = {}

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    last_accel_path = save_checkpoint(accelerator, transformer, args.output_dir, global_step, epoch, step_within_epoch, args.use_lora, args.save_accelerator_state, last_accel_path, cumulative_flops=_flops_cumulative if args.measure_flops else None)

                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    accelerator.print(f"[eval] Running periodic eval at step={global_step}...")
                    eval_t0 = time.perf_counter()
                    accelerator.wait_for_everyone()
                    eval_metrics = run_periodic_eval(
                        accelerator=accelerator,
                        args=args,
                        transformer=transformer,
                        models=models,
                        build_eval_pipe=build_eval_pipe,
                        global_step=global_step,
                        model_dtype=model_dtype,
                    )
                    accelerator.wait_for_everyone()
                    eval_sec_total += time.perf_counter() - eval_t0

                    # Always include training time in eval output, even when
                    # eval_every_steps is not a multiple of log_every.
                    eval_metrics["time/total_sec"] = time.perf_counter() - total_start
                    eval_metrics["time/train_sec"] = eval_metrics["time/total_sec"] - eval_sec_total

                    if accelerator.is_main_process and len(eval_metrics) > 0:
                        metric_parts = [f"{k}={v:.4f}" for k, v in eval_metrics.items()]
                        accelerator.print("[eval] " + " ".join(metric_parts))
                        # Accumulate total token usage across all eval rounds
                        total_prompt_tokens += int(eval_metrics.get("eval/total_prompt_tokens", 0.0))
                        total_completion_tokens += int(eval_metrics.get("eval/total_completion_tokens", 0.0))
                    log_dict.update(eval_metrics)
                    # Write cumulative token totals AFTER update so they don't get overwritten by per-round values
                    log_dict["eval/total_prompt_tokens"] = float(total_prompt_tokens)
                    log_dict["eval/total_completion_tokens"] = float(total_completion_tokens)

                if log_dict and args.use_wandb:
                    accelerator.log(log_dict, step=global_step)

                if global_step >= args.max_train_steps:
                    accelerator.print(f"[train] Reached max_train_steps={args.max_train_steps}")
                    break

        if global_step >= args.max_train_steps:
            break

    if args.save_final_checkpoint:
        accelerator.print(f"[train] Saving final checkpoint at step={global_step}")
        last_accel_path = save_checkpoint(accelerator, transformer, args.output_dir, global_step, epoch if args.num_epochs > 0 else 0, step_within_epoch if args.num_epochs > 0 else 0, args.use_lora, args.save_accelerator_state, last_accel_path, checkpoint_name="checkpoint-final", cumulative_flops=_flops_cumulative if args.measure_flops else None)
    if args.use_wandb:
        accelerator.end_training()
    accelerator.print(f"[train] Training complete. final_step={global_step}")
    accelerator.print(
        f"[train] Total judge token usage: prompt_tokens={total_prompt_tokens}, "
        f"completion_tokens={total_completion_tokens}"
    )
    if last_accel_path:
        accelerator.print(f"[train] To resume training, load from: {last_accel_path}")


# Local indirection so `configs.resolve_from_scratch_config` and
# `transformers.get_scheduler` are imported lazily (only when from-scratch / a
# scheduler is actually built) — keeping train_loop importable standalone
# (no backend `configs` dependency at import time).
def resolve_from_scratch_config_local(value):
    from configs import resolve_from_scratch_config
    return resolve_from_scratch_config(value)


def get_scheduler_local(*args, **kwargs):
    from transformers import get_scheduler
    return get_scheduler(*args, **kwargs)
