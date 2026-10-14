#!/usr/bin/env bash
set -euo pipefail

# Generate the full TinyStories dataset with shard pack mode.
# Run from project root.
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="$(pwd)"

python "${PROJECT_ROOT}/src/text_dataset/generate_tinystories.py" \
  --splits train,validation \
  --num_workers 8 \
  --pack \
  --shard_size 50000 \
  --output_dir "${PROJECT_ROOT}/data/tinystories/full_data_shard"