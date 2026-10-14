# Sudoku: single-task SelfCond adapter (experiment 06)

DiffThinker Sudoku: complete the 81-cell grid. One SelfCond
Qwen-Image-Edit-2511 LoRA adapter (`diffthinker_large_sudoku`), trained 5
epochs to `checkpoint-final/lora`. FLOPs are measured and recorded for
provenance but never paired against a vanilla counterpart.

- Table columns: `tab:diffthinker_large`, `sudoku_45`, `sudoku_40`,
  `sudoku_35`, and `sudoku_30` (givens count)
- Test set: 400 instances (`formatted/Sudoku/metadata_test.csv`)
- GT: `data/diffthinker/EndoCoT/Sudoku`
- Scorer: `diffthinker_eval/sudoku`, vLLM + Qwen3-VL-8B per-cell OCR (GPU).
  Requires `QWEN3VL_CKPT` (path to Qwen3-VL-8B) and `VLLM_TP=1` at scoring.

## Prerequisites

Same base as the representative leaf [`../vsp/`](../vsp/README.md): eight CUDA
GPUs, `requirements.txt`, `.env`, and the EndoCoT/DiffThinker source
(`DIFFTHINKER_SOURCE`, or the provisioner's HuggingFace assembly). Scoring
also needs the GPU OCR environment: vLLM, `qwen_vl_utils`, and
`QWEN3VL_CKPT`. `OPENAI_API_*` is only used by the diagnostic periodic eval in
stage 03.

## How to run

```bash
cd scripts/06_diffthinker_large_scale/task_specific/sudoku
export QWEN3VL_CKPT=/path/to/Qwen3-VL-8B
bash run.sh --dry-run            # resolve the config; print its path and the stage names
bash run.sh --profile paper      # stages 01 to 06
```

Outputs land under `outputs/06_diffthinker_large_scale/sudoku/`.

## Stages

Identical six-stage pipeline as [`../vsp/`](../vsp/README.md); the stage
scripts are shared and read this leaf's `config.yaml`:

```
01_prepare_data -> 02_measure_flops -> 03_train_selfcond -> 04_evaluate -> 05_score -> 06_summarize
```

Training saves and evaluates every 2500 steps; stage 04 generates at 50 NFE,
seed 0, judge off; stage 05 scores with the Sudoku OCR parser; stage 06 writes
`by_difficulty.csv`, which the parent `summarize.sh` consumes. Stage 03's
periodic eval is diagnostic only.

## Documentation

- [`docs/reproduction.md`](../../../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../../../docs/flops_protocol.md): FLOPs
  measurement and pairing protocol
- [`docs/data.md`](../../../../docs/data.md#diffthinker-provenance):
  DiffThinker data provenance
- [`../README.md`](../README.md): task table shared by the five leaves
- [`../../README.md`](../../README.md): group overview and contracts
