#!/usr/bin/env bash
set -euo pipefail

# Baseline vanilla training on image-conditioned A5 line traces. Run from project root.
export PYTHONUNBUFFERED=1
export EXPERIMENT_METHOD="vanilla"

PROJECT_ROOT="$(pwd)"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-base-4B}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/a5_line_trace_image/full_data_shard/train}"
METADATA_CSV="${METADATA_CSV:-${DATA_ROOT}/metadata.csv}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-${PROJECT_ROOT}/data/a5_line_trace_image/full_data_shard/val}"
EVAL_METADATA_CSV="${EVAL_METADATA_CSV:-${EVAL_DATA_ROOT}/metadata.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/a5_line_trace_image/baseline_100M}"

python "${PROJECT_ROOT}/src/image_only/train.py" \
  --metadata_csv "${METADATA_CSV}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --transformer_model_name_or_path "${MODEL_ID}" \
  --vae_model_name_or_path "${MODEL_ID}" \
  --from_scratch_config 100M \
  --learning_rate 1e-4 \
  --num_epochs 20 \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --num_workers 8 \
  --preload_shards_count 2 \
  --save_steps 1000 \
  --log_every 50 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 1000 \
  --eval_every_steps 1000 \
  --eval_max_samples 20 \
  --eval_num_inference_steps 100 \
  --eval_metadata_csv "${EVAL_METADATA_CSV}" \
  --eval_data_root "${EVAL_DATA_ROOT}" \
  --eval_model_name_or_path "${MODEL_ID}" \
  --eval_disable_judge \
  --eval_num_visualize 3 \
  --use_wandb \
  --wandb_project a5-line-trace-image \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"}

