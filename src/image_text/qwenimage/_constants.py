"""QwenImage-Edit default model id + subfolder constants."""

# Base edit model. Verify the exact repo id at download time (Edit vs Edit-Plus,
# "2509"/"2511" variant, etc.). The base QwenImageEditPipeline consumes this.
DEFAULT_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_TRANSFORMER_SUBFOLDER = "transformer"
DEFAULT_VAE_SUBFOLDER = "vae"
DEFAULT_TEXT_ENCODER_SUBFOLDER = "text_encoder"
DEFAULT_TOKENIZER_SUBFOLDER = "tokenizer"
DEFAULT_PROCESSOR_SUBFOLDER = "processor"
