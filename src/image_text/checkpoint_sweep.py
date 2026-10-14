"""Evaluate every checkpoint produced by one training run.

The backend entry points reuse their sibling ``train.py`` parser and model
loader.  This module supplies the backbone-agnostic parts: checkpoint
discovery, in-place weight/adapter loading, Accelerate setup, distributed
periodic evaluation, W&B logging, and an on-disk summary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class CheckpointSpec:
    path: Path
    name: str
    global_step: int
    cumulative_flops: float | None


def discover_checkpoints(output_dir: str, *, use_lora: bool) -> list[CheckpointSpec]:
    """Return complete ``checkpoint-*`` directories in training-step order."""
    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"--output_dir does not exist or is not a directory: {root}")

    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    )
    if not candidates:
        raise ValueError(f"No checkpoint-* directories found under: {root}")

    model_subdir = "lora" if use_lora else "transformer"
    checkpoints: list[CheckpointSpec] = []
    invalid: list[str] = []
    for path in candidates:
        state_path = path / "training_state.pt"
        weights_path = path / model_subdir
        missing = []
        if not state_path.is_file():
            missing.append("training_state.pt")
        if not weights_path.is_dir():
            missing.append(f"{model_subdir}/")
        if missing:
            invalid.append(f"{path.name} (missing {', '.join(missing)})")
            continue

        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if "global_step" not in state:
            invalid.append(f"{path.name} (training_state.pt has no global_step)")
            continue
        global_step = int(state["global_step"])
        cumulative_flops = state.get("cumulative_flops")
        checkpoints.append(
            CheckpointSpec(
                path=path,
                name=path.name,
                global_step=global_step,
                cumulative_flops=(
                    float(cumulative_flops) if cumulative_flops is not None else None
                ),
            )
        )

    if invalid:
        expected_flag = "--use_lora" if use_lora else "full-parameter mode (no --use_lora)"
        details = "\n  - ".join(invalid)
        raise ValueError(
            f"Found incomplete or incompatible checkpoint directories for {expected_flag}:\n"
            f"  - {details}"
        )
    if not checkpoints:
        raise ValueError(f"No usable checkpoint-* directories found under: {root}")

    # checkpoint-final commonly shares a step with the last numeric checkpoint;
    # place it after the numeric save so both are evaluated deterministically.
    checkpoints.sort(
        key=lambda checkpoint: (
            checkpoint.global_step,
            checkpoint.name == "checkpoint-final",
            checkpoint.name,
        )
    )
    return checkpoints


def _safetensor_shards(model_dir: Path) -> list[Path]:
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError(f"Multiple safetensors index files found in {model_dir}: {index_files}")
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_files[0]}")
        shards = [model_dir / name for name in sorted(set(weight_map.values()))]
    else:
        shards = sorted(model_dir.glob("*.safetensors"))

    if not shards:
        raise ValueError(f"No safetensors weights found in transformer checkpoint: {model_dir}")
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise ValueError(f"Missing safetensors shard(s) referenced by checkpoint: {missing}")
    return shards


@torch.inference_mode()
def _load_full_weights_in_place(transformer, model_dir: Path) -> None:
    """Strictly copy a full checkpoint into the existing model, shard by shard."""
    from safetensors.torch import load_file

    state = transformer.state_dict()
    expected = set(state)
    loaded_names: set[str] = set()

    for shard_path in _safetensor_shards(model_dir):
        shard = load_file(str(shard_path), device="cpu")
        duplicate = loaded_names.intersection(shard)
        if duplicate:
            raise ValueError(
                f"Duplicate tensor keys across checkpoint shards in {model_dir}: "
                f"{sorted(duplicate)[:8]}"
            )
        unexpected = set(shard).difference(expected)
        if unexpected:
            raise ValueError(
                f"Unexpected tensor keys in {shard_path}: {sorted(unexpected)[:8]}"
            )

        for name, source in shard.items():
            target = state[name]
            if source.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {name} in {model_dir}: "
                    f"checkpoint={tuple(source.shape)}, model={tuple(target.shape)}"
                )
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        loaded_names.update(shard)
        del shard

    missing = expected.difference(loaded_names)
    if missing:
        raise ValueError(f"Checkpoint {model_dir} is missing tensor keys: {sorted(missing)[:8]}")


def _load_lora_checkpoint(transformer, lora_dir: Path, adapter_name: str):
    """Attach the first adapter or atomically replace the active sweep adapter."""
    from peft import PeftModel

    if not isinstance(transformer, PeftModel):
        transformer = PeftModel.from_pretrained(
            transformer,
            str(lora_dir),
            adapter_name=adapter_name,
            is_trainable=False,
        )
        return transformer, adapter_name

    previous_names = list(transformer.peft_config)
    transformer.load_adapter(str(lora_dir), adapter_name=adapter_name, is_trainable=False)
    transformer.set_adapter(adapter_name)
    for previous_name in previous_names:
        if previous_name != adapter_name:
            transformer.delete_adapter(previous_name)
    return transformer, adapter_name


def _runtime_dtype(accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    if accelerator.device.type != "cuda":
        dtype = torch.float32
    return dtype


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(temporary, path)


def run_checkpoint_sweep(
    args,
    *,
    extra_args: dict[str, Any],
    load_models: Callable,
    build_eval_pipe: Callable,
) -> None:
    """Load and evaluate every checkpoint in ``args.output_dir``."""
    # Imported lazily so discovery helpers stay usable in lightweight test
    # environments that do not have the training stack installed.
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        InitProcessGroupKwargs,
        set_seed,
    )
    from periodic_eval import run_periodic_eval
    from train_loop import resolve_from_scratch_config_local

    setattr(args, "experimental", extra_args)
    if args.eval_metadata_csv is None:
        raise ValueError("--eval_metadata_csv is required for checkpoint sweep evaluation")
    if args.eval_num_inference_steps <= 0:
        raise ValueError("--eval_num_inference_steps must be > 0")
    if args.from_scratch_config is not None and args.use_lora:
        raise ValueError("--from_scratch_config and --use_lora are mutually exclusive")
    if args.preload_shards and args.preload_shards_count > 0:
        raise ValueError(
            "--preload_shards and --preload_shards_count > 0 are mutually exclusive"
        )
    args.from_scratch_config_dict = (
        resolve_from_scratch_config_local(args.from_scratch_config)
        if args.from_scratch_config is not None
        else None
    )

    checkpoints = discover_checkpoints(args.output_dir, use_lora=args.use_lora)
    if args.eval_output_dir is None:
        args.eval_output_dir = os.path.join(args.output_dir, "checkpoint_eval")
    eval_root = Path(args.eval_output_dir)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=args.find_unused_parameters
    )
    nccl_timeout_sec = int(os.environ.get("NCCL_TIMEOUT_SEC", "3600"))
    init_pg_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=nccl_timeout_sec)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
    )
    accelerator.print(
        f"[checkpoint-sweep] Accelerator initialized: device={accelerator.device}, "
        f"mixed_precision={accelerator.mixed_precision}, "
        f"processes={accelerator.num_processes}"
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        eval_root.mkdir(parents=True, exist_ok=True)
        _write_json(eval_root / "checkpoint_sweep_args.json", vars(args))
    accelerator.wait_for_everyone()

    if args.use_wandb:
        wandb_init_kwargs: dict[str, Any] = {}
        if args.wandb_run_name is not None:
            wandb_init_kwargs["name"] = f"{args.wandb_run_name}-checkpoint-sweep"
        if args.wandb_entity is not None:
            wandb_init_kwargs["entity"] = args.wandb_entity
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": wandb_init_kwargs} if wandb_init_kwargs else None,
        )
        if accelerator.is_main_process:
            tracker = accelerator.get_tracker("wandb", unwrap=True)
            tracker.define_metric("checkpoint/global_step")
            tracker.define_metric("eval/*", step_metric="checkpoint/global_step")
            tracker.define_metric("bl/*", step_metric="checkpoint/global_step")
        accelerator.print(
            f"[checkpoint-sweep] W&B enabled: project={args.wandb_project}, "
            f"run_name={wandb_init_kwargs.get('name')}"
        )

    model_dtype = _runtime_dtype(accelerator)
    accelerator.print(f"[checkpoint-sweep] Runtime model dtype: {model_dtype}")
    accelerator.print(
        f"[checkpoint-sweep] Found {len(checkpoints)} checkpoint(s): "
        + ", ".join(checkpoint.name for checkpoint in checkpoints)
    )

    models = load_models(args, model_dtype, accelerator)
    transformer = models.transformer
    transformer.requires_grad_(False)
    if args.compile:
        accelerator.print(
            "[checkpoint-sweep] --compile is a training option and is ignored; "
            "checkpoint weights must remain directly replaceable between evaluations."
        )

    summary: list[dict[str, Any]] = []
    summary_path = eval_root / "checkpoint_sweep_summary.json"
    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        for checkpoint_index, checkpoint in enumerate(checkpoints):
            accelerator.wait_for_everyone()
            accelerator.print(
                f"[checkpoint-sweep] [{checkpoint_index + 1}/{len(checkpoints)}] "
                f"Loading {checkpoint.name} (global_step={checkpoint.global_step})"
            )

            if args.use_lora:
                # Alternate names so the new adapter can be loaded before the old
                # one is removed, leaving no interval without an active adapter.
                adapter_name = f"checkpoint_sweep_{checkpoint_index % 2}"
                transformer, _ = _load_lora_checkpoint(
                    transformer,
                    checkpoint.path / "lora",
                    adapter_name,
                )
            else:
                _load_full_weights_in_place(
                    transformer,
                    checkpoint.path / "transformer",
                )

            transformer.requires_grad_(False)
            transformer.to(device=accelerator.device, dtype=model_dtype)
            transformer.eval()
            models.transformer = transformer
            accelerator.wait_for_everyone()

            metrics = run_periodic_eval(
                accelerator=accelerator,
                args=args,
                transformer=transformer,
                models=models,
                build_eval_pipe=build_eval_pipe,
                global_step=checkpoint.global_step,
                model_dtype=model_dtype,
                output_name=checkpoint.name,
            )
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                total_prompt_tokens += int(metrics.get("eval/total_prompt_tokens", 0))
                total_completion_tokens += int(
                    metrics.get("eval/total_completion_tokens", 0)
                )
                record = {
                    "checkpoint": checkpoint.name,
                    "checkpoint_path": str(checkpoint.path),
                    "global_step": checkpoint.global_step,
                    "cumulative_flops": checkpoint.cumulative_flops,
                    "metrics": metrics,
                }
                summary.append(record)
                _write_json(summary_path, summary)

                if args.use_wandb:
                    wandb_metrics: dict[str, Any] = dict(metrics)
                    wandb_metrics.update(
                        {
                            "checkpoint/name": checkpoint.name,
                            "checkpoint/index": checkpoint_index,
                            "checkpoint/global_step": checkpoint.global_step,
                            "checkpoint_sweep/total_prompt_tokens": total_prompt_tokens,
                            "checkpoint_sweep/total_completion_tokens": total_completion_tokens,
                        }
                    )
                    if checkpoint.cumulative_flops is not None:
                        wandb_metrics["checkpoint/cumulative_flops"] = (
                            checkpoint.cumulative_flops
                        )
                    accelerator.log(wandb_metrics, step=checkpoint_index)

                metric_parts = [
                    f"{key}={value:.4f}"
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                ]
                accelerator.print(
                    f"[checkpoint-sweep] Completed {checkpoint.name}: "
                    + " ".join(metric_parts)
                )
            # Keep non-main ranks alive until the main process has flushed the
            # summary and W&B row, especially after the final checkpoint.
            accelerator.wait_for_everyone()
    finally:
        if args.use_wandb:
            accelerator.end_training()

    accelerator.print(
        f"[checkpoint-sweep] Finished {len(checkpoints)} checkpoint(s). "
        f"Summary: {summary_path}"
    )
