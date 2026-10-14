# Experiment 01: line tracing

Line tracing is a pure-visual sequential reasoning task with five-color paths
and a fixed number of steps (m=50). This leaf trains a from-scratch FLUX.2
transformer (preset `9M_LONG`, no text encoder; code in `src/image_only/`)
twice, once as VanillaFM and once as SelfCondFM, pairs the two checkpoints at
matched training FLOPs, and scores the selected pair on 20 fixed held-out
instances.

- Metric: `max_correct_prefix_steps.mean` (LCP) over 20 fixed instances at
  100 NFE
- Index entry: `docs/reproduction.md` (paper figures and artifacts index)

## Prerequisites

- Linux with four CUDA GPUs (`train.num_processes: 4`); the paper runs used
  A100 80GB.
- `pip install -r requirements.txt` at the repository root.
- `.env` copied from `.env.example`. The dataset repository is
  [alfalfa-w/self-conditioning-deep-reasoning-data](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data); its downloads do not require
  a token. Maintainers can use `DATA_SOURCE_MODE=inherit_original` and
  `ORIGINAL_REPO_ROOT` to read the original checkout.
- No API judge is used: scoring is pure-visual (`eval.judge: disabled`).

Training at a glance (all values come from `config.yaml`): four processes,
per-device batch 8, no accumulation (effective batch 32), learning rate 2e-3
with a constant-with-warmup schedule, 10 epochs, bf16, seed 1201, and save and
eval every 1000 steps. Evaluation runs at 100 NFE with seed 1000.

Data provisioning: the config declares `source_mode: hf`, and
`01_prepare_data.sh` downloads and verifies the pinned original release.
Maintainers can read the original data with `DATA_SOURCE_MODE=inherit_original`
and `ORIGINAL_REPO_ROOT` set; every mode verifies sha256 and row counts. The
`regenerate` mode reruns the generator with the config geometry, but it does
not match the historical generation seeds or fonts byte-for-byte.

## How to run

```bash
cd scripts/01_line_tracing
bash run.sh --dry-run            # resolve the config; print its path and the stage names (no GPU, no network)
bash run.sh --profile paper      # stages 01 to 07
bash run.sh --stage 05_match_checkpoints --stage 06_evaluate   # selected stages, in argument order
bash run_multiseed.sh            # three-seed sweep (fig:multi_seed_line_tracing)
bash generate_examples.sh        # qualitative panels; run after 06_evaluate
```

`run.sh` validates the config with `scripts/common/preflight.sh` before any
stage runs. Every stage script is independently callable and resolves
`scripts/common/env.sh` on its own.

Multi-seed: `run_multiseed.sh` provisions the data and measures the FLOPs
rates once, because both are seed-invariant, and runs them automatically when
their outputs are absent. It then runs train, pair, evaluate, and summarize
for each seed in `multiseed.seeds` (1201, 1202, 1203) under a per-seed config
in `outputs/01_line_tracing/multiseed/<seed>/`, with pairing redone per seed.
The aggregate `multiseed_summary.json` reports the mean CTD and the sample
standard deviation per method; the script exits nonzero when any seed's scores
are missing.

Examples: `generate_examples.sh` consumes the selected pair (`pairs.json`) and
the predictions already produced by stage 06; it regenerates neither. It picks
three exemplars from the evaluation, preferring instances where
SelfCondFM is fully correct and VanillaFM is not, composes a four-panel strip
(input, reference, vanilla, selfcond) at the native 3216x80 geometry, records
the per-sample LCP of both methods, and writes `examples/manifest.json` beside
the evaluation outputs. The historical cherry-pick IDs listed in the config are
recorded as `source_missing`: their original predictions, scores, and
checkpoints were not recovered, and no similar-looking sample is substituted.

## Stages

Output paths are relative to `outputs/01_line_tracing/`.

| Stage | Does | Produces |
| --- | --- | --- |
| `01_prepare_data` | Provision and verify the fixed m=50 dataset; freeze the manifest | `data/line_tracing/` with `manifest.json` |
| `02_measure_flops` | 100-micro-batch measurement run per method | `flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | From-scratch `9M_LONG`, `EXPERIMENT_METHOD=vanilla` | `vanilla/checkpoint-*/` |
| `04_train_selfcond` | From-scratch `9M_LONG`, `EXPERIMENT_METHOD=proj_condition` | `selfcond/checkpoint-*/` |
| `05_match_checkpoints` | `flops apply`, then `audit`, then `match` (shared protocol) | stamped `training_state.pt`, `audit.json`, `pairs.json` |
| `06_evaluate` | Inference at 100 NFE plus pure-visual CTD scoring for the selected pair | per-instance results under `eval/` |
| `07_summarize` | Aggregate metrics, budget, and provenance | `summary.json`, `curves.svg` |

## Contracts

- `save_steps == eval_every_steps` (both 1000), so every periodic-eval step has
  a recoverable checkpoint, as `docs/flops_protocol.md` requires.
- FLOPs measurement and training are decoupled: stages 03 and 04 read no rate.
  Stage 05 stamps each checkpoint with
  `cumulative_flops = per_micro_batch_flops * global_step * grad_accum * num_processes`,
  using the measured rate of that checkpoint's own method. Rates are never
  shared between the two methods.
- Pairing is anchored at the SelfCondFM step `checkpoints.paper_selfcond_step: 53000`.
  The matcher takes the highest SelfCond candidate and pairs it with the
  nearest Vanilla checkpoint satisfying `F_vanilla >= F_selfcond`, with no gap
  threshold. It steps back to a lower SelfCond candidate only when no Vanilla
  checkpoint reaches the current candidate's FLOPs. Accuracy never participates
  in the selection.
- The measured selfcond-to-vanilla per-micro-batch ratio (about 1.123 here) is
  recorded for provenance; no `[2.0, 3.5]` band is checked. See
  `docs/flops_protocol.md`.
- Historical endpoints: Vanilla@60000 and SelfCond@53000.

## Outputs

New runs write under `outputs/01_line_tracing/`, and the multi-seed sweep under
`outputs/01_line_tracing/multiseed/<seed>/`. The frozen paper numbers live
under `results/reference/` and are never overwritten by a new run.

## Documentation

- [`docs/reproduction.md`](../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../docs/flops_protocol.md): FLOPs measurement
  and pairing
- [`../README.md`](../README.md): orchestration conventions shared by all leaves
