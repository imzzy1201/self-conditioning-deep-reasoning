# Experiment 05: controlled DiffThinker tasks

This group reproduces the five-task controlled comparison on the DiffThinker
evaluation set with the FLUX.2-klein-base-4B backbone (code in
`src/image_text/flux2/`, frozen Qwen3 text encoder, frozen VAE, LoRA rank and
alpha 32, learning rate 1e-4, global batch 16, 50 NFE inference). Each task
trains a VanillaFM adapter and a SelfCondFM adapter (`proj_condition`) from the
same seed 1201, pairs the two at matched training FLOPs, and evaluates the
selected pair on the frozen DiffThinker test manifest.

## Tasks

| Task dir | Label | Difficulties | Test instances | Judge | Extra dependencies |
| --- | --- | --- | --- | --- | --- |
| `vsp/` | VSP-Super | 16, 32 | 200 | gymnasium FrozenLake walk (CPU) | `gymnasium` |
| `maze/` | Maze | 8, 16, 32 | 300 | walk-the-maze `.txt` verifier (CPU) | none |
| `tsp/` | TSP | 12, 15, 18 | 300 | cv2 HSV parse, edge-overlap tolerance 0.5 (CPU) | none |
| `sudoku/` | Sudoku | 45, 40, 35, 30 (givens) | 400 | vLLM Qwen3-VL per-cell digit OCR (GPU) | vLLM, `qwen_vl_utils`, `QWEN3VL_CKPT` |
| `jigsaw/` | Jigsaw | 2x2, 3x3, 4x4 | 300 | vLLM Qwen3-VL per-piece label OCR (GPU) | vLLM, `qwen_vl_utils`, `QWEN3VL_CKPT` |

Each task directory contains `config.yaml`, `run.sh` (seven
stages), `generate_examples.sh`, and its own `README.md`. The stage scripts are
byte-identical across the five leaves; task identity lives in `config.yaml`.
The group-level `summarize.sh` aggregates the five leaves into the frozen
reference rows under `results/reference/diffthinker_controlled/`.

## Per-task stages

| Stage | Does |
| --- | --- |
| `01_prepare_data` | Verify and freeze the external EndoCoT/DiffThinker data provenance (`data_manifest.json`) |
| `02_measure_flops` | 100-micro-batch measurement run per method, producing `flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | VanillaFM, max 48000 steps, save == eval == 2500 |
| `04_train_selfcond` | SelfCondFM, max 16000 steps (maze: 23400), save == eval == 1000, `proj_condition` |
| `05_match_checkpoints` | `flops_cli` apply, audit, match, anchored at `paper_selfcond_step: 16000` |
| `06_evaluate` | Standalone generation (judge disabled, 50 NFE, seed 0) plus the task's external scorer |
| `07_summarize` | Per-task `metrics.csv` (overall accuracy, full manifest denominator) and `summary.json` |
| `generate_examples` | Qualitative panels for `fig:qual_<task>`, written to `examples/diffthinker/<task>/manifest.json` |

## Prerequisites

- Linux with four CUDA GPUs (`train.num_processes: 4`); the paper runs used
  A100 80GB.
- `pip install -r requirements.txt` at the repository root, plus `gymnasium`
  for VSP and vLLM with `qwen_vl_utils` for Sudoku and Jigsaw.
- `.env` copied from `.env.example`.
- External data: the EndoCoT/DiffThinker source tree. Set `DIFFTHINKER_SOURCE`
  to your checkout, or leave it unset and let the provisioner assemble the tree
  from pinned HuggingFace and upstream sources (needs network and git). Stage
  01 calls `scripts/common/provision_diffthinker_data.sh`, which mirrors the
  source into `data/diffthinker/`, verifies pinned metadata row counts and
  sha256, and stops with a report when the data cannot be sourced or verified.
  It never fabricates data. See
  [`docs/data.md`](../../docs/data.md#diffthinker-provenance).
- Sudoku and Jigsaw scoring also needs `QWEN3VL_CKPT` (optional
  `VLLM_TP` / `JUDGE_GPU`).

## How to run

```bash
cd scripts/05_diffthinker_controlled/vsp
bash run.sh --dry-run                # resolve the config; print its path and the stage names
bash 01_prepare_data.sh              # provision and verify the external data, freeze provenance
bash run.sh --profile paper          # stages 01 to 07
bash generate_examples.sh            # qualitative panels, after 06_evaluate

cd .. && bash summarize.sh           # after all five leaves wrote metrics.csv
```

Repeat the leaf commands for `maze`, `tsp`, `sudoku`, and `jigsaw`. Each leaf
README documents its task-specific judge and run instructions.

## Contracts

- `save_steps == eval_every_steps > 0` per run, enforced by preflight and read
  from `train.{save,eval}_steps` for Vanilla and
  `train.selfcond_{save,eval}_steps` for SelfCond. Vanilla saves on the source
  eval grid (2500), the grid the paper's Vanilla curve is sampled on, and
  `paper_vanilla_step: 40000` lands on it. SelfCond keeps its source cadence
  1000/1000.
- Every declared `checkpoints.paper_{vanilla,selfcond}_step` must be `final` or
  an exact multiple of the matching save cadence, or preflight fails: the
  trainer never writes a periodic checkpoint off its save grid.
- Pair selection is FLOPs-driven only, anchored at `paper_selfcond_step: 16000`:
  the highest SelfCond step whose nearest Vanilla satisfies
  `F_vanilla >= F_selfcond` (non-strict), with a per-step fallback and no gap
  threshold. The algorithm lives in `src/experiment_tools/flops.py` and is
  called through `flops_cli match`; it is never re-implemented here. Accuracy
  plays no role in selection.
- `num_epochs` is a ceiling; `--max_train_steps` (48000 vanilla, 16000
  selfcond) is the budget.
- SelfCond switches: `EXPERIMENT_METHOD=proj_condition AUX_LOSS=1
  AUX_LOSS_WEIGHT=0.5 STOP_GRAD_PROJ=1 ZERO_PROJ_WHEN_ABSENT=0
  PROJ_TIME_OFFSET=-10` plus `--condition_time_offset 10`. Training and
  inference must use the same `ZERO_PROJ_WHEN_ABSENT` (0 drops the empty
  projection slot).
- The cross-task average is the equal-weight mean of the five task overalls.
  Raw samples are never merged, because the denominators differ
  (200/300/300/400/300).
- Accuracy = PASS / manifest instances (full denominator). MISSING results and
  parse errors count as incorrect.

## Outputs

- Per-task runs write under `outputs/05_diffthinker_controlled/<task>/`.
- `summarize.sh` writes the frozen reference rows under
  `results/reference/diffthinker_controlled/` (`metrics.csv`, `pairs.json`,
  `manifest.json`).
- Qualitative panels land under `examples/diffthinker/<task>/`.

## Documentation

- [`docs/reproduction.md`](../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index (E05 / D05 / C06)
- [`docs/data.md`](../../docs/data.md#diffthinker-provenance):
  EndoCoT/DiffThinker data provenance
- [`docs/flops_protocol.md`](../../docs/flops_protocol.md): measurement and
  pairing
- [`../README.md`](../README.md): orchestration conventions
