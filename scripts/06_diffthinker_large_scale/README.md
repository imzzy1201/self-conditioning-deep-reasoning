# Experiment 06: large-scale DiffThinker tasks and the unified adapter

This group runs single-model SelfCondFM adapters on Qwen-Image-Edit-2511 and
compares them with published baselines. There is no VanillaFM/SelfCondFM
training pair here: each task gets one SelfCond adapter (C07, under
`task_specific/`) and one adapter is trained jointly on four tasks (C08, under
`unified/`, excluding Jigsaw). Every run trains 5 epochs and produces
`checkpoint-final/lora`. Training FLOPs are measured and recorded for
provenance, but they are never `match`-paired against the external baselines.

- Backbone: `src/image_text/qwenimage/` (Qwen/Qwen-Image-Edit-2511, 5D VAE and
  frozen Qwen2.5-VL text encoder). The token order is
  `[target, projection, condition]`, with the target at index 0. The SelfCond
  method is `proj_condition`.

## Layout

| Path | Purpose |
| --- | --- |
| `task_specific/<task>/` | The five single-task adapters: VSP, Maze, TSP, Sudoku, Jigsaw (C07) |
| `unified/` | Four-task joint training on VSP + Maze + TSP + Sudoku, without Jigsaw (C08) |
| `summarize.sh` | Group aggregator: writes `results/reference/diffthinker_large_scale/{task_specific,unified,external_baselines}.csv` and `jigsaw/{by_difficulty,external_baselines}.csv` |

Each leaf (`task_specific/<task>/` and `unified/`) contains a
`config.yaml` plus six stages orchestrated by `run.sh`:

| Stage | Does |
| --- | --- |
| `01_prepare_data` | Freeze train/test metadata and GT provenance (record-only) |
| `02_measure_flops` | Single-model SelfCond FLOPs measurement run (recorded; no pairing) |
| `03_train_selfcond` | Train 5 epochs, producing `checkpoint-final/lora` |
| `04_evaluate` | Generate from the frozen final adapter at 50 NFE, seed 0, judge off |
| `05_score` | Per-task DiffThinker parse and eval, producing per-instance JSON |
| `06_summarize` | Per-difficulty counts with the full denominator, producing `by_difficulty.csv` |

## Prerequisites

- Linux with eight CUDA GPUs (`train.num_processes: 8`); the source runs used
  eight processes at per-device batch 1 (global batch 8).
- `pip install -r requirements.txt` at the repository root, and `.env` copied
  from `.env.example`.
- External data: set `DIFFTHINKER_SOURCE` to your EndoCoT/DiffThinker checkout,
  or leave it unset and let the provisioner assemble the tree from pinned
  HuggingFace and upstream sources (needs network and git). Every
  `01_prepare_data.sh` calls `scripts/common/provision_diffthinker_data.sh`,
  which mirrors the source into `data/diffthinker/` and verifies pinned
  metadata row counts and sha256, stopping with a report when the data cannot
  be sourced or verified. The unified leaf's `unified/eval_unified.csv` is
  repo-local and derived (no upstream union eval CSV exists; the source run had
  periodic eval off); it is diagnostic only. See
  [`docs/data.md`](../../docs/data.md#diffthinker-provenance).
- `task_specific` leaves: the stage-03 periodic `visual_full` eval is judge-on
  and needs `OPENAI_API_*` credentials (diagnostic only).
- Sudoku and Jigsaw scoring: vLLM, `qwen_vl_utils`, `QWEN3VL_CKPT`, and
  `VLLM_TP=1`. VSP scoring: `gymnasium`.

## How to run

```bash
cd scripts/06_diffthinker_large_scale/task_specific/vsp
bash run.sh --dry-run                # resolve the config; print its path and the stage names
bash run.sh --profile paper          # stages 01 to 06

cd ../../unified
bash run.sh --profile paper

cd ..
bash summarize.sh                    # after the leaves produced by_difficulty.csv
```

## Contracts

- Task-specific leaves run with `save_steps == eval_every_steps > 0`, enforced
  by preflight. The unified leaf keeps periodic eval disabled
  (`eval_every_steps: 0`), matching its source run. Periodic eval is diagnostic
  only either way: the final table numbers come from `04_evaluate` plus
  `05_score` on `checkpoint-final/lora`.
- SelfCond switches: `EXPERIMENT_METHOD=proj_condition AUX_LOSS=1
  AUX_LOSS_WEIGHT=0.5 STOP_GRAD_PROJ=1 ZERO_PROJ_WHEN_ABSENT=0`. On QwenImage,
  `PROJ_TIME_OFFSET` and `--condition_time_offset` have no numeric effect
  (token positions derive from the `img_shapes` list order); they are kept for
  router parity.
- The main table is 12 difficulty columns (VSP 16/32, Maze 8/16/32, TSP
  12/15/18, Sudoku 45/40/35/30), averaged with equal weight over the columns
  (task-specific 90.0, unified 90.2). Jigsaw is a separate 3-column table
  (2x2/3x3/4x4, mean 96.0).
- Local rows are marked `local_reproduction`; published rows are
  `published_reference` (EndoCoT v1 Table 1 for the main table, DiffThinker v1
  Table 1 for Jigsaw). A missing task or difficulty is reported `incomplete`
  and never 0-filled.
- `checkpoint-final` must read its step from `training_state.pt`; the
  directory name alone is not proof of the final step.

## Outputs

Runs write under `outputs/06_diffthinker_large_scale/`; the frozen reference
numbers live under `results/reference/diffthinker_large_scale/`.

## Documentation

- [`docs/reproduction.md`](../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index (E06/E07, D05,
  C07/C08)
- [`docs/flops_protocol.md`](../../docs/flops_protocol.md): FLOPs measurement
  and pairing protocol
- [`docs/data.md`](../../docs/data.md#diffthinker-provenance):
  source tree, pinned CSVs, derived `eval_unified.csv`
- [`../README.md`](../README.md): orchestration conventions
