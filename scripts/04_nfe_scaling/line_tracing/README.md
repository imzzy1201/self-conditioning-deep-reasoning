# Experiment 04: NFE scaling, line tracing

This leaf sweeps the number of function evaluations (NFE 1 to 50) and records
transformer-forward FLOPs over 200 held-out m=50 line-tracing instances,
evaluating the SelfCond/Vanilla pair that experiment 01 selected (C01). No
training happens here and the pair is never re-selected.

- Metric: `max_correct_prefix_steps.mean` (LCP) over 200 fixed instances (D04)
- Index entry: `docs/reproduction.md` (paper figures and artifacts index)

## Prerequisites

- A completed experiment 01 run. Stage 02 fails unless
  `outputs/01_line_tracing/pairs.json` is matched and both selected checkpoints
  carry a `transformer/` subdirectory.
- One CUDA GPU for the inference sweep (`eval.device: cuda`, bf16). This leaf
  trains nothing.
- `pip install -r requirements.txt` at the repository root, and `.env` copied
  from `.env.example`.
- Data: the HF release includes the original NFE subset. Maintainers
  can use `DATA_SOURCE_MODE=inherit_original` and `ORIGINAL_REPO_ROOT` to read
  the original checkout.

## How to run

```bash
cd scripts/04_nfe_scaling/line_tracing
bash run.sh --dry-run                  # resolve the config; print its path and the stage names
bash run.sh --profile paper            # stages 01 to 04
bash run.sh --stage 03_sweep_nfe       # re-run only the sweep
bash run.sh --stage 04_summarize       # re-aggregate existing cells
```

Stage scripts are independently callable; each resolves
`scripts/common/env.sh` on its own and loads `.env`.

## Stages

Output paths are relative to `outputs/04_nfe_scaling/line_tracing/`.

| Stage | Does | Produces |
| --- | --- | --- |
| `01_prepare_data` | Provision and freeze the 200-instance m=50 `nfe_eval` split | `data/line_tracing/nfe_eval/` with `manifest.json` |
| `02_locate_checkpoints` | Locate experiment 01's `pairs.json`, verify each selected checkpoint's `transformer/` subdirectory, freeze a local pair copy | frozen `pairs.json` |
| `03_sweep_nfe` | `src/experiment_tools/eval_nfe_scaling.py` per method: inference at every NFE from 1 to 50, CTD scoring, and per-cell inference FLOPs | per-method results under `eval/` |
| `04_summarize` | Same-cells aggregation for both paper axes | `cells.csv`, `inference_flops.json`, `summary.json` |

## Contracts

- Pair reuse: the whole sweep consumes experiment 01's `pairs.json` as-is
  (`outputs.pair_source`). Stage 02 only locates and freezes the pair; it never
  calls `flops match`. Re-selection happens once, in `scripts/01_line_tracing`,
  never per sample or per NFE.
- Two axes, one cell table: `fig:nfe_scaling` (x = NFE 1 to 50) and
  `fig:inference_flops` (x = per-cell transformer-forward FLOPs) are both built
  from the `cells.csv` that stage 04 writes. Neither axis triggers a
  re-evaluation.
- NFE is not FLOPs: the NFE axis is the `--num_inference_steps` sweep, while
  the FLOPs axis is the measured per-sample transformer-forward count recorded
  by `--measure_inference_flops` (FlopCounterMode; VAE and scheduler excluded).
  The first SelfCondFM step carries no projection token and later steps do, so
  the cost per step is not constant.
- Fixed noise: `eval.noise_seed` is pinned in the config (0). Sample `i` uses
  `torch.Generator(noise_seed + i)` at every NFE, identical across both
  methods.
- Incomplete cells are reported, not hidden: stage 04 reports `n_expected`,
  `n_scored`, and the missing count per cell and marks missing cells
  `incomplete`. It never shrinks the denominator.

## Outputs

New runs write under `outputs/04_nfe_scaling/line_tracing/`. The frozen
reference artifacts live under `results/reference/nfe_scaling/line_tracing/`
and are never overwritten by a new run.

## Documentation

- [`docs/reproduction.md`](../../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../../docs/flops_protocol.md):
  inference-FLOPs counting and pairing protocol
- [`../README.md`](../README.md): this group's overview
- [`../../README.md`](../../README.md): orchestration conventions shared by all leaves
