# Unified: four-task joint SelfCond adapter (C08), no Jigsaw

One SelfCond Qwen-Image-Edit-2511 LoRA adapter trained jointly on VSP, Maze,
TSP, and Sudoku (Jigsaw excluded). The paper's `tab:diffthinker_large` unified
row reports all 12 difficulty columns from this single model; the parent
`summarize.sh` reads this leaf's `by_difficulty.csv` into the `ours_unified`
row.

- id: `diffthinker_large_unified`
- Joint train set: `data/diffthinker/formatted/train_unified.csv`, 182,400
  rows (Maze 75,000 + Sudoku 40,000 + TSP 30,000 + VSP 37,400); row count and
  sha256 pinned by `scripts/common/provision_diffthinker_data.sh`
- Training: eight processes at per-device batch 1 (global batch 8), learning
  rate 1e-4 with a linear schedule and 1000 warmup steps, 5 epochs, bf16, seed
  1201, save every 11400 steps, periodic eval disabled (`eval_every_steps: 0`)
  to match the source run
- Evaluation and scoring fan out per task: stage 04 writes
  `<eval_dir>/{vsp,maze,tsp,sudoku}/predictions`, stage 05 writes
  `<score_dir>/per_instance_{vsp,maze,tsp,sudoku}.json`, and stage 06
  aggregates the 12 buckets into this leaf's `by_difficulty.csv` under
  `outputs/06_diffthinker_large_scale/unified/`

## Prerequisites

- Linux with eight CUDA GPUs (`train.num_processes: 8`).
- `pip install -r requirements.txt` at the repository root, plus `gymnasium`
  for VSP scoring and vLLM with `qwen_vl_utils` and `QWEN3VL_CKPT` for Sudoku
  scoring. `.env` copied from `.env.example`.
- The external EndoCoT/DiffThinker source: set `DIFFTHINKER_SOURCE` to your
  checkout, or leave it unset and let the provisioner assemble the tree from
  pinned HuggingFace and upstream sources (needs network and git).
- No `OPENAI_API_*` credentials are needed: periodic eval is disabled and
  stage 04 runs with the judge off.

## How to run

```bash
cd scripts/06_diffthinker_large_scale/unified
bash run.sh --dry-run            # resolve the config; print its path and the stage names
bash run.sh --profile paper      # stages 01 to 06
```

## Stages

| Stage | Does |
| --- | --- |
| `01_prepare_data` | Verify the joint train metadata and the four per-task eval/GT roots, producing `data_manifest.json` |
| `02_measure_flops` | 100-micro-batch SelfCond FLOPs measurement run, producing `flops.json` (recorded; no pairing) |
| `03_train_selfcond` | Joint SelfCond training, 5 epochs, save every 11400, periodic eval off |
| `04_evaluate` | Per-task generation from the frozen final adapter (50 NFE, seed 0, judge off) |
| `05_score` | Per-task DiffThinker scoring, producing `per_instance_<task>.json` |
| `06_summarize` | Per-(task, difficulty) aggregation, producing the unified `by_difficulty.csv` |

## Documentation

- [`docs/reproduction.md`](../../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../../docs/flops_protocol.md): FLOPs
  measurement protocol
- [`docs/data.md`](../../../docs/data.md#diffthinker-provenance):
  pinned source CSVs and the derived `eval_unified.csv`
- [`../task_specific/vsp/README.md`](../task_specific/vsp/README.md):
  representative single-task leaf (identical stage scripts)
- [`../README.md`](../README.md): group overview and contracts
