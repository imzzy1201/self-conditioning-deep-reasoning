#!/usr/bin/env bash
set -euo pipefail

# Generate 200K TinyStories text-to-image samples (no edit image) + validation set.
# prompt = stored as text, target_image = rendered response, no condition image.
# Output as plain PNG files (no shard pack).
# Run from project root.
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="$(pwd)"

python "${PROJECT_ROOT}/src/text_dataset/generate_tinystories.py" \
  --splits train,validation \
  --max_samples 300000 \
  --num_workers 8 \
  --no_edit_image \
  --image_width 256 \
  --image_height 256 \
  --font_size 13 \
  --output_dir "${PROJECT_ROOT}/data/tinystories/300k_no_edit"
