# 05_diffthinker_controlled/maze: Maze (walk-the-maze path tracing)

Controlled FLUX.2 comparison of VanillaFM and SelfCondFM on DiffThinker's Maze
test set (difficulties 8, 16, and 32, 300 instances). The model draws the red
path from the start to the exit of a maze. The judge is the walk-the-maze
`.txt` verifier, CPU only.

- Paper: `tab:diffthinker` Maze 64.3 (Vanilla) / 70.7 (SelfCond),
  `paper_reported` reference values
- Train raw: `EndoCoT-Data/Maze` with `formatted/Maze/metadata_train.csv`
- Eval GT: `EndoCoT/Maze` with `formatted/Maze/metadata_test.csv`
- Pairing anchors: SelfCond 16000, Vanilla 40000

## Prerequisites

Four CUDA GPUs, `pip install -r requirements.txt`, `.env` copied from
`.env.example`, and the external EndoCoT/DiffThinker source
(`DIFFTHINKER_SOURCE`, or the provisioner's HuggingFace assembly). The group
[`README.md`](../README.md) has the full list and the data-provisioning
details.

## How to run

```bash
cd scripts/05_diffthinker_controlled/maze
bash run.sh --dry-run            # resolve the config; print its path and the stage names
bash run.sh --profile paper      # stages 01 to 07
bash generate_examples.sh        # fig:qual_maze panels; run after 06_evaluate
```

## Stages

| Stage | Does |
| --- | --- |
| `01_prepare_data` | Verify and freeze the external data provenance (`data_manifest.json`) |
| `02_measure_flops` | 100-micro-batch measurement run per method, producing `flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | VanillaFM, max 48000 steps, save == eval == 2500 |
| `04_train_selfcond` | SelfCondFM, max 23400 steps, save == eval == 1000, `proj_condition` |
| `05_match_checkpoints` | `flops_cli` apply, audit, match (`F_vanilla >= F_selfcond`, no gap threshold) |
| `06_evaluate` | Generation (50 NFE, seed 0, judge disabled) plus the Maze external scorer |
| `07_summarize` | `metrics.csv` and `summary.json` |
| `generate_examples` | `fig:qual_maze` panels, written to `examples/diffthinker/maze/manifest.json` |

Outputs land under `outputs/05_diffthinker_controlled/maze/`.

## Contracts

- `save_steps == eval_every_steps` per run (Vanilla 2500, SelfCond 1000),
  enforced by preflight.
- Pairing is FLOPs-driven only and anchored at SelfCond@16000; accuracy never
  participates. The algorithm lives in `src/experiment_tools/flops.py` and is
  described in [`docs/flops_protocol.md`](../../../docs/flops_protocol.md).
- The full contract list (cadences, anchor grid, method switches, averaging) is
  identical for all five tasks and lives in the group
  [`README.md`](../README.md).

## Documentation

- [`docs/reproduction.md`](../../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../../docs/flops_protocol.md): FLOPs
  measurement and pairing
- [`docs/data.md`](../../../docs/data.md#diffthinker-provenance):
  external data provenance
- [`../README.md`](../README.md): group overview and shared contracts
