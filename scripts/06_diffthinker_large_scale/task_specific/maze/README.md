# Maze: single-task SelfCond adapter (experiment 06)

DiffThinker Maze: draw the red path from the start to the exit. One SelfCond
Qwen-Image-Edit-2511 LoRA adapter (`diffthinker_large_maze`), trained 5 epochs
to `checkpoint-final/lora`. FLOPs are measured and recorded for provenance but
never paired against a vanilla counterpart.

- Table columns: `tab:diffthinker_large`, `maze_8`, `maze_16`, and `maze_32`
- Test set: 300 instances (`formatted/Maze/metadata_test.csv`)
- GT: `data/diffthinker/EndoCoT/Maze`
- Scorer: `diffthinker_eval/maze`, a CPU path-walk

## Prerequisites

Same as the representative leaf [`../vsp/`](../vsp/README.md): eight CUDA GPUs,
`requirements.txt`, `.env`, and the EndoCoT/DiffThinker source
(`DIFFTHINKER_SOURCE`, or the provisioner's HuggingFace assembly). No GPU OCR
service is needed for Maze scoring; `OPENAI_API_*` is only used by the
diagnostic periodic eval in stage 03.

## How to run

```bash
cd scripts/06_diffthinker_large_scale/task_specific/maze
bash run.sh --dry-run            # resolve the config; print its path and the stage names
bash run.sh --profile paper      # stages 01 to 06
```

Outputs land under `outputs/06_diffthinker_large_scale/maze/`.

## Stages

Identical six-stage pipeline as [`../vsp/`](../vsp/README.md); the stage
scripts are shared and read this leaf's `config.yaml`:

```
01_prepare_data -> 02_measure_flops -> 03_train_selfcond -> 04_evaluate -> 05_score -> 06_summarize
```

Training saves and evaluates every 4688 steps; stage 04 generates at 50 NFE,
seed 0, judge off; stage 05 scores with the Maze parser; stage 06 writes
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
