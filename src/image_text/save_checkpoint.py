"""Shared checkpoint saver (backbone-agnostic)."""

import os
import shutil

import torch
from accelerate import Accelerator


def save_checkpoint(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    output_dir: str,
    global_step: int,
    epoch: int,
    step_within_epoch: int,
    use_lora: bool,
    save_accelerator_state: bool = False,
    last_accel_path: str | None = None,
    checkpoint_name: str | None = None,
    cumulative_flops: float | None = None,
):
    ckpt_name = checkpoint_name or f"checkpoint-{global_step}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    accelerator.print(f"[checkpoint] Start saving: step={global_step}, dir={ckpt_dir}")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
        model_to_save = accelerator.unwrap_model(transformer)
        model_dir = "lora" if use_lora else "transformer"
        model_path = os.path.join(ckpt_dir, model_dir)
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        model_to_save.save_pretrained(model_path)
        state = {"global_step": global_step, "epoch": epoch, "step_within_epoch": step_within_epoch}
        if cumulative_flops is not None:
            state["cumulative_flops"] = float(cumulative_flops)
        torch.save(state, state_path)
        accelerator.print(f"[checkpoint] Saved model weights to: {model_path}")
        accelerator.print(f"[checkpoint] Saved training state to: {state_path}")

    new_accel_path: str | None = None
    if save_accelerator_state:
        new_accel_path = os.path.join(ckpt_dir, "accelerator")
        accelerator.save_state(new_accel_path)
        accelerator.wait_for_everyone()
        accelerator.print(f"[checkpoint] Saved accelerator state to: {new_accel_path}")
        if last_accel_path and accelerator.is_main_process and os.path.exists(last_accel_path):
            shutil.rmtree(last_accel_path, ignore_errors=True)
            accelerator.print(f"[checkpoint] Deleted old accelerator state: {last_accel_path}")

    accelerator.print(f"[checkpoint] Finished checkpoint-{global_step}")
    return new_accel_path
