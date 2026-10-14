# Experiment 02: long multiplication

Long multiplication renders a whole column-arithmetic worksheet (5 to 7 digit
factors) as a single image, and the model must generate the completed
worksheet as an image. This leaf fine-tunes FLUX.2-klein-base-4B with LoRA
(rank and alpha 32, no text encoder; code in `src/image_only/`) as VanillaFM
and as SelfCondFM, pairs the two checkpoints at matched training FLOPs, and
scores the selected pair with a whole-worksheet API judge.

- Metric: `judge_num_correct / 100`, the full-set accuracy with all 100 test
  rows in the denominator. Valid-only accuracy is reported separately and is
  never substituted for it.
- Index entry: `docs/reproduction.md` (paper figures and artifacts index)

## Prerequisites

- Linux with four CUDA GPUs (`train.num_processes: 4`); the paper runs used
  A100 80GB.
- `pip install -r requirements.txt` at the repository root.
- `.env` copied from `.env.example`. The dataset repository
  [alfalfa-w/self-conditioning-deep-reasoning-data](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data) needs no token for downloads
  (`HF_ENDPOINT` defaults to the mirror `https://hf-mirror.com`). Maintainers on
  the original cluster can use `DATA_SOURCE_MODE=inherit_original` with
  `ORIGINAL_REPO_ROOT` instead.
- API judge: scoring calls the model in `eval.judge_model` (currently
  `qwen3.8-flash`) through `OPENAI_API_KEY` / `OPENAI_API_BASE` / `OPENAI_MODEL`
  from `.env`. Evaluation is not offline.

Training at a glance (all values come from `config.yaml`): effective batch 8
(per-device 1 x accumulation 2 x 4 processes); worksheet sizes vary, so the
per-device batch is 1 and `torch.compile` is not used. Learning rate 1e-4 with
a constant-with-warmup schedule, bf16, seed 1201. VanillaFM trains 15 epochs
with save and eval every 12500 steps; SelfCondFM trains 5 epochs with save and
eval every 5000 steps (`train.selfcond_*`). Evaluation runs at 50 NFE with seed
1000 over the 100 test worksheets.

Data provisioning: the config ships `source_mode: hf`, so stage 01 downloads
the frozen worksheets (200,000 train, 100 test, 5 to 7 digits) and verifies
sha256 and row counts before linking them into `data/long_multiplication/`.
`regenerate` mode reruns the generator but does not match the historical
generation byte-for-byte.

## How to run

```bash
cd scripts/02_long_multiplication
bash run.sh --dry-run            # resolve the config; print its path and the stage names (no GPU, no network)
bash run.sh --profile paper      # stages 01 to 07
bash run.sh --stage 05_match_checkpoints --stage 06_evaluate   # selected stages, in argument order
bash run_multiseed.sh            # three-seed sweep (fig:multi_seed_long_multiplication)
bash generate_examples.sh        # qualitative panels; run after 06_evaluate
```

`run.sh` validates the config with `scripts/common/preflight.sh` before any
stage runs. Every stage script is independently callable and resolves
`scripts/common/env.sh` on its own.

Multi-seed: `run_multiseed.sh` provisions the data and measures the FLOPs
rates once (both are seed-invariant, and missing ones are produced
automatically), then runs train, pair, evaluate, and summarize for each seed in
`multiseed.seeds` (1201, 1202, 1203) under a per-seed config in
`outputs/02_long_multiplication/multiseed/<seed>/`. The sweep is anchored at
SelfCond@50000 (`multiseed.paper_selfcond_step`), a smaller budget than the
single-seed anchor SelfCond@125000; do not mix the two.

Examples: `generate_examples.sh` consumes the selected pair (`pairs.json`) and
stage 06's judge results (`judge_results.jsonl`); it regenerates neither. It
composes four-panel strips (input, reference, vanilla, selfcond) at the native
worksheet size, records the per-sample judge vote, and writes
`examples/manifest.json`. The historical cherry-pick IDs in the config (5, 10,
12) are recorded as `source_missing`: their original predictions, scores, and
checkpoints were not recovered, and no similar-looking sample is substituted.

## Stages

Output paths are relative to `outputs/02_long_multiplication/`.

| Stage | Does | Produces |
| --- | --- | --- |
| `01_prepare_data` | Provision and verify the 200k train and 100 test worksheets; freeze the manifest | `data/long_multiplication/` with `manifest.json` |
| `02_measure_flops` | 100-micro-batch measurement run per method (LoRA) | `flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | LoRA on the pretrained 4B transformer, `EXPERIMENT_METHOD=vanilla`, 15 epochs | `vanilla/checkpoint-*/` |
| `04_train_selfcond` | Same backbone, `EXPERIMENT_METHOD=proj_condition`, 5 epochs | `selfcond/checkpoint-*/` |
| `05_match_checkpoints` | `flops apply`, then `audit`, then `match` (shared protocol) | stamped `training_state.pt`, `audit.json`, `pairs.json` |
| `06_evaluate` | Inference at 50 NFE plus the whole-worksheet judge for the selected pair | per-instance results under `eval/` |
| `07_summarize` | Aggregate judge metrics, budget, and provenance | `summary.json`, `rows.csv` |

## Contracts

- Each run keeps `save_steps == eval_every_steps > 0`: Vanilla 12500, SelfCond
  5000, expressed as `train.selfcond_{save,eval}_steps` for the second method.
  `preflight.sh` checks both cadences.
- Full-set accuracy puts all 100 test rows in the denominator. Judge
  disagreements, unparseable votes, and API failures contribute neither to the
  numerator nor to the valid denominator; they are reported as `num_error` and
  `num_unknown`.
- Pairing is anchored at `checkpoints.paper_selfcond_step: 125000` (multi-seed:
  50000). The historical pair Vanilla@312500 / SelfCond@125000 carries a FLOPs
  gap of about 3.08%, which is why the pair is re-selected under the shared
  protocol in `docs/flops_protocol.md`.
- Judge protocol: the whole-worksheet prompt `src/image_only/eval_prompt_longmul.txt`,
  k=2 votes, strict mode.
- `preload_shards_count: 0` matches the source runs. Any value between 0 and
  the shard count would enable partial preloading and silently disable
  shuffling.

## Outputs

New runs write under `outputs/02_long_multiplication/`, and the multi-seed
sweep under `outputs/02_long_multiplication/multiseed/<seed>/`. The frozen
paper numbers live under `results/reference/` and are never overwritten by a
new run.

## Documentation

- [`docs/reproduction.md`](../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../docs/flops_protocol.md): FLOPs measurement
  and pairing
- [`../README.md`](../README.md): orchestration conventions shared by all leaves
