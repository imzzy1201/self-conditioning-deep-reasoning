"""FLUX.2-Klein default model id + subfolder constants."""

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_TRANSFORMER_SUBFOLDER = "transformer"
DEFAULT_VAE_SUBFOLDER = "vae"
DEFAULT_TEXT_ENCODER_SUBFOLDER = "text_encoder"
DEFAULT_TOKENIZER_SUBFOLDER = "tokenizer"
# FLUX.2 has no Qwen2VLProcessor; this constant exists only for parity with the
# qwenimage backend so cli_args.add_common_train_args can be shared.
DEFAULT_PROCESSOR_SUBFOLDER = "processor"
