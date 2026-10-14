# 05_diffthinker_controlled/sudoku: Sudoku (81-cell digit OCR)

Controlled FLUX.2 comparison of VanillaFM and SelfCondFM on DiffThinker's
Sudoku test set (givens-count difficulties 45, 40, 35, and 30, 400 instances).
The model completes the 81-cell grid. The judge is vLLM Qwen3-VL per-cell digit
OCR on GPU (`needs_gpu_ocr: true`; set `QWEN3VL_CKPT`, optional `VLLM_TP` /
`JUDGE_GPU`).

- Paper: `tab:diffthinker` Sudoku 60.5 (Vanilla) / 80.0 (SelfCond),
  `paper_reported` reference values
- Train raw: `EndoCoT-Data/Sudoku` with `formatted/Sudoku/metadata_train.csv`
- Eval GT: `EndoCoT/Sudoku` with `formatted/Sudoku/metadata_test.csv`
- Pairing anchors: SelfCond 16000, Vanilla 40000

## Prerequisites

Four CUDA GPUs, `pip install -r requirements.txt`, `.env` copied from
`.env.example`, and the external EndoCoT/DiffThinker source
(`DIFFTHINKER_SOURCE`, or the provisioner's HuggingFace assembly). Scoring
also needs vLLM, `qwen_vl_utils`, and `QWEN3VL_CKPT`. The group
[`README.md`](../README.md) has the full list and the data-provisioning
details.

## How to run

```bash
cd scripts/05_diffthinker_controlled/sudoku
bash run.sh --dry-run            # resolve the config; print its path and the stage names
bash run.sh --profile paper      # stages 01 to 07
bash generate_examples.sh        # fig:qual_sudoku panels; run after 06_evaluate
```

## Stages

| Stage | Does |
| --- | --- |
| `01_prepare_data` | Verify and freeze the external data provenance (`data_manifest.json`) |
| `02_measure_flops` | 100-micro-batch measurement run per method, producing `flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | VanillaFM, max 48000 steps, save == eval == 2500 |
| `04_train_selfcond` | SelfCondFM, max 16000 steps, save == eval == 1000, `proj_condition` |
| `05_match_checkpoints` | `flops_cli` apply, audit, match (`F_vanilla >= F_selfcond`, no gap threshold) |
| `06_evaluate` | Generation (50 NFE, seed 0, judge disabled) plus the Sudoku OCR scorer |
| `07_summarize` | `metrics.csv` and `summary.json` |
| `generate_examples` | `fig:qual_sudoku` panels, written to `examples/diffthinker/sudoku/manifest.json` |

Outputs land under `outputs/05_diffthinker_controlled/sudoku/`.

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
