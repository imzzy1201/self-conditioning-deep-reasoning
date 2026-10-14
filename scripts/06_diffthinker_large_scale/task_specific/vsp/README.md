# VSP (FrozenLake): single-task SelfCond adapter (experiment 06)

DiffThinker "Visual State-space Planning": given a FrozenLake map and a
starting cell, draw the action path (red line). One SelfCond
Qwen-Image-Edit-2511 LoRA adapter (`diffthinker_large_vsp`), trained 5 epochs
to `checkpoint-final/lora`. Experiment 06 trains a single SelfCond model per
task: FLOPs are measured and recorded for provenance but never paired against
a vanilla counterpart.

- Table columns: `tab:diffthinker_large`, `vsp_16` and `vsp_32` (difficulties
  16/32)
- Test set: 200 instances (`formatted/VSP/metadata_test.csv`)
- GT: `data/diffthinker/EndoCoT/FrozenLake`
- Scorer: `diffthinker_eval/vsp`, a CPU FrozenLake walk (needs `gymnasium`)

## Prerequisites

- Linux with eight CUDA GPUs (`train.num_processes: 8`; global batch 8 at
  per-device batch 1).
- `pip install -r requirements.txt` at the repository root, plus `gymnasium`
  for scoring, and `.env` copied from `.env.example`.
- The external EndoCoT/DiffThinker source: set `DIFFTHINKER_SOURCE` to your
  checkout, or leave it unset and let the provisioner assemble the tree from
  pinned HuggingFace and upstream sources (needs network and git).
- `OPENAI_API_*` credentials for the stage-03 periodic eval, which is judge-on
  and diagnostic only.

## How to run

```bash
cd scripts/06_diffthinker_large_scale/task_specific/vsp
bash run.sh --dry-run            # resolve the config; print its path and the stage names (no GPU, no network)
bash run.sh --profile paper      # stages 01 to 06
```

Outputs land under `outputs/06_diffthinker_large_scale/vsp/`.

## Stages

`run.sh` orchestrates six stages; each reads this leaf's `config.yaml`.

| Stage | Does |
| --- | --- |
| `01_prepare_data` | Provision and verify the DiffThinker mirror; freeze train/test metadata and GT provenance into `data_manifest.json` (record-only) |
| `02_measure_flops` | 100-micro-batch SelfCond FLOPs measurement run, producing `flops.json` (record-only; no pairing) |
| `03_train_selfcond` | Train SelfCond 5 epochs, save and eval every 2338 steps, producing `checkpoint-final/lora` |
| `04_evaluate` | Generate from the frozen final adapter at 50 NFE, seed 0, judge off, producing `<eval_dir>/predictions` |
| `05_score` | `eval_diffthinker.py --task vsp`, producing `per_instance_vsp.json` |
| `06_summarize` | Per-difficulty aggregation, producing `by_difficulty.csv` (consumed by the parent `summarize.sh`) |

Stage 03 also runs a source-aligned periodic eval: judge-on `visual_full`
(k=2, strict) over the full 200-sample test set, bl metrics, and
eval-before-training. It is diagnostic only and needs `OPENAI_API_*`; the
reported numbers come from stages 04 and 05 on `checkpoint-final/lora`.

## Contracts

- SelfCond switches: `EXPERIMENT_METHOD=proj_condition`, `AUX_LOSS=1`,
  `AUX_LOSS_WEIGHT=0.5`, `STOP_GRAD_PROJ=1`, `ZERO_PROJ_WHEN_ABSENT=0`.
- `save_steps == eval_every_steps` (2338), enforced by preflight.
- Global batch 8; learning rate 1e-4, linear schedule, 1000 warmup steps,
  bf16, seed 1201.

## Documentation

- [`docs/reproduction.md`](../../../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../../../docs/flops_protocol.md): FLOPs
  measurement and pairing protocol
- [`docs/data.md`](../../../../docs/data.md#diffthinker-provenance):
  DiffThinker data provenance
- [`../README.md`](../README.md): task table shared by the five leaves
- [`../../README.md`](../../README.md): group overview and contracts
