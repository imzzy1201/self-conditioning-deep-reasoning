# Experiment 03: long text rendering (TinyStories)

Long text rendering asks the model to draw a long passage of text as an image.
Each sample stores a TinyStories story as a text prompt, and the target
image renders the longest wrapped prefix that fits a fixed canvas. This leaf
fine-tunes FLUX.2-klein-base-4B with LoRA (rank and alpha 32) through the
`src/image_text/flux2` trainer, which uses the frozen Qwen3 text encoder
(guidance_scale=1.0). It trains VanillaFM and SelfCondFM, pairs the two
checkpoints at matched training FLOPs, and scores the selected pair by OCR with
a character-error-rate judge.

- Metric: `cer_mean`, the character error rate averaged over valid OCR samples
  only, and `perfect/100`, the exact-match count. Paper values: Vanilla 0.046,
  SelfCond 0.005; perfect 13 and 44 out of 100.
- Index entry: `docs/reproduction.md` (paper figures and artifacts index);
  pairing in `docs/flops_protocol.md`

## Prerequisites

- Linux with four CUDA GPUs (`train.num_processes: 4`); the paper runs used
  A100 80GB.
- `pip install -r requirements.txt` at the repository root.
- `.env` copied from `.env.example`. The dataset repository
  [alfalfa-w/self-conditioning-deep-reasoning-data](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data) needs no token for downloads
  (`HF_ENDPOINT` defaults to the mirror `https://hf-mirror.com`). Maintainers on
  the original cluster can use `DATA_SOURCE_MODE=inherit_original` with
  `ORIGINAL_REPO_ROOT` instead.
- OCR judge: scoring calls `qwen3.8-flash` (enable_thinking, thinking budget
  5000, up to 3 retries) through `OPENAI_API_KEY` / `OPENAI_API_BASE` /
  `OPENAI_MODEL` from `.env`. Evaluation is not offline.

Training at a glance (all values come from `config.yaml`): effective batch 16
(per-device 1 x accumulation 4 x 4 processes), so 300,000 rows give 18,750
steps per epoch. VanillaFM trains 12 epochs (225,000 steps) with save and eval
every 5000 steps; SelfCondFM trains 4 epochs (75,000 steps) with save and eval
every 2000 steps (`train.selfcond_*`). Learning rate 1e-4, bf16, seed 1201.
Evaluation runs at 50 NFE over the first 100 test rows, with per-sample noise
`seed + row_index` reproduced through `eval.noise_shards: 4`.

Data provisioning: the config ships `source_mode: hf`, so stage 01 downloads
the frozen TinyStories set (300k no-edit train rows plus the test split),
verifies sha256 and row counts, and links them into
`data/long_text_rendering/`. `regenerate` mode reruns the generator but does
not match the historical generation byte-for-byte.

## How to run

```bash
cd scripts/03_long_text_rendering
bash run.sh --dry-run            # resolve the config; print its path and the stage names (no GPU, no network)
bash run.sh --profile paper      # stages 01 to 07
bash run.sh --stage 05_match_checkpoints --stage 06_evaluate   # selected stages, in argument order
bash run_multiseed.sh            # three-seed sweep (extension; not a paper figure)
bash generate_examples.sh        # qualitative panels; run after 06_evaluate
```

`run.sh` validates the config with `scripts/common/preflight.sh` before any
stage runs. Every stage script is independently callable and resolves
`scripts/common/env.sh` on its own.

Multi-seed: `run_multiseed.sh` runs train, pair, evaluate, and summarize for
each seed in `multiseed.seeds` (1201, 1202, 1203) under per-seed configs in
`outputs/03_long_text_rendering/multiseed/<seed>/`, anchored at the main
protocol's SelfCond@74000. This sweep is an extension; the paper reports no
multi-seed numbers for this task.

Examples: `generate_examples.sh` consumes the selected pair (`pairs.json`) and
stage 06's OCR results (`cer_results.jsonl`); it regenerates neither. Because
this is a no-edit task there is no input image, so each exemplar is a
three-panel strip (reference, vanilla, selfcond) at the native render size,
with per-sample CER recorded for both methods, plus `examples/manifest.json`.
The historical cherry-pick IDs in the config (7, 18, 20) are recorded as
`source_missing`: their original predictions, scores, and checkpoints were not
recovered, and no similar-looking sample is substituted.

## Stages

Output paths are relative to `outputs/03_long_text_rendering/`.

| Stage | Does | Produces |
| --- | --- | --- |
| `01_prepare_data` | Provision and verify the TinyStories 300k no-edit train and test splits; freeze the manifest | `data/long_text_rendering/` with `manifest.json` |
| `02_measure_flops` | 100-micro-batch measurement run per method (LoRA) | `flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | LoRA, 12 epochs, `EXPERIMENT_METHOD=vanilla` | `vanilla/checkpoint-*/` |
| `04_train_selfcond` | LoRA, 4 epochs, `EXPERIMENT_METHOD=proj_condition` | `selfcond/checkpoint-*/` |
| `05_match_checkpoints` | `flops apply`, then `audit`, then `match` (shared protocol) | stamped `training_state.pt`, `audit.json`, `pairs.json` |
| `06_evaluate` | Standalone inference plus the OCR/CER judge for the selected pair | per-instance results under `eval/` |
| `07_summarize` | Aggregate CER, budget, provenance, and curve series | `summary.json`, `cer_curves.json`, `rows.csv` |

## Trainer invocation

All training and evaluation stages launch the image_text trainer by absolute
path; this leaf does not use the legacy `src/image_only` entry points:

```bash
export PYTHONPATH="$REPO_ROOT/src/image_text:$REPO_ROOT/src"
accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  "$REPO_ROOT/src/image_text/flux2/train.py" --metadata_csv ... --data_root <SPLIT_DIR> ...
```

- `--data_root` is the split directory (`data/long_text_rendering/train`),
  because the generator writes metadata image paths relative to the split
  directory (PNG mode).
- SelfCond environment contract, verified against
  `src/image_text/experiment_router.py` and
  `src/image_text/flux2/proj_condition_loss.py`:
  `EXPERIMENT_METHOD=proj_condition AUX_LOSS=1 AUX_LOSS_WEIGHT=0.5
  STOP_GRAD_PROJ=1 ZERO_PROJ_WHEN_ABSENT=0 PROJ_TIME_OFFSET=-10` plus
  `--condition_time_offset 10`. This is the image_text contract, not the legacy
  `src/image_only` PROJ_TIME_OFFSET contract.
- `06_evaluate.sh` uses `src/image_text/flux2/evaluation.py` with
  `--judge_mode cer`, `--lora_path <ckpt>/lora`, and `--experiment_method`, and
  it exports `PROJ_TIME_OFFSET` and `ZERO_PROJ_WHEN_ABSENT` so inference
  matches training.
- `REPO_ROOT` and the runtime roots (`DATA_ROOT`, `OUTPUT_ROOT`,
  `REFERENCE_ROOT`) come from `scripts/common/env.sh`, which derives the root
  from its own location, loads `.env`, and provides `resolve_path`. Stage
  scripts do not recompute any root locally.

## Contracts

- Each run keeps `save_steps == eval_every_steps > 0`: Vanilla 5000, SelfCond
  2000 (`train.selfcond_{save,eval}_steps`; `preflight.sh` checks both). The
  anchors land on grid points: Vanilla@190000 = 38 x 5000 and SelfCond@74000 =
  37 x 2000.
- Pairing is anchored at `checkpoints.paper_selfcond_step: 74000`; historical
  endpoints are Vanilla@190000 and SelfCond@74000.
- `cer_mean` averages over valid OCR samples only. OCR errors are counted
  separately as `num_error` and never folded into the mean; `perfect_count/100`
  is the exact-match count. An `_all`-imputed CER is not a paper metric and is
  not computed here.

## Outputs

New runs write under `outputs/03_long_text_rendering/`, and the multi-seed
sweep under `outputs/03_long_text_rendering/multiseed/<seed>/`. The frozen
paper numbers live under `results/reference/` and are never overwritten by a
new run. The `reference` block of `config.yaml` records the frozen paper values
(CER 0.0461806 vanilla and 0.0050397 selfcond, perfect 13 and 44) for
provenance only; no stage re-measures them.

## Documentation

- [`docs/reproduction.md`](../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../docs/flops_protocol.md): FLOPs measurement
  and pairing
- [`../README.md`](../README.md): orchestration conventions shared by all leaves
