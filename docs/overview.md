# Repository overview

This repository reproduces the main experiments of the paper
*Self-Conditioning Enables Deep Sequential Reasoning in Flow Matching*. It
trains flow-matching image models (FLUX.2-klein and Qwen-Image) on
visual-reasoning tasks, pairs VanillaFM against SelfCondFM checkpoints at
matched training FLOPs, and scores them with task-specific evaluators.

Each experiment is a leaf under `scripts/`: one `config.yaml`
(the single source of truth for that experiment) plus `run.sh` and numbered
stage scripts. This guide covers the layout, pipeline, data and environment
setup, and quickstart for all 15 leaves. The top-level
[`README.md`](../README.md) is the entry point.

## Layout

```text
├── README.md
├── requirements.txt              # runtime deps (torch, diffusers, transformers, accelerate, peft, ...)
├── .env.example                  # copy to .env; API keys + external data-source roots
├── docs/
│   ├── overview.md               # this guide
│   ├── reproduction.md           # environment, path contract, acceptance, figure/artifact index
│   ├── flops_protocol.md         # FLOPs measurement + pairing algorithm
│   ├── codebase.md               # directory-by-directory codebase map
│   └── data.md                   # datasets: hosting, pins, loading, DiffThinker provenance
├── src/
│   ├── image_only/               # pure-visual FLUX.2 training, sampling, diagnostics
│   ├── image_text/               # flux2 / qwenimage / eval_external trainers + evaluators
│   ├── experiment_tools/         # FLOPs, pairing, eval summarization
│   ├── analysis/                 # plotting, tables, example layout
│   ├── *_dataset/                # synthetic + text dataset generators
│   └── */scripts/                # historical dataset entry scripts (provenance)
├── scripts/
│   ├── common/                   # env.sh, read_config.py, preflight.sh, flops.sh (manual wrapper), provisioners
│   ├── 01_line_tracing/          # three synthetic main tasks,
│   ├── 02_long_multiplication/   #   each with run_multiseed.sh
│   ├── 03_long_text_rendering/   #   and generate_examples.sh
│   ├── 04_nfe_scaling/           # NFE-scaling sweep (line tracing)
│   ├── 05_diffthinker_controlled/   # five FLUX.2 LoRA tasks (+ group summarize.sh)
│   └── 06_diffthinker_large_scale/  # task_specific/* + unified (Qwen-Image; + group summarize.sh)
├── data/                         # runtime, git-ignored (provisioned per docs/data.md)
└── outputs/                      # runtime, git-ignored
```

Every leaf holds `README.md`, `config.yaml`, `run.sh`, and numbered stage
scripts. Within `05_diffthinker_controlled/*` and
`06_diffthinker_large_scale/task_specific/*`, corresponding stage scripts are
byte-identical across task directories; each task's identity lives in its
config. Package-local `src/*/scripts/` records historical dataset provenance;
the runnable experiment pipeline lives in the repo-root `scripts/` tree.

## Anatomy of an experiment

The three synthetic tasks and five controlled tasks use these seven stages.
Run commands from each leaf; config paths resolve against the repository root.

| Stage | What it does | Produces |
| --- | --- | --- |
| `01_prepare_data` | Provision and verify data (sha256 + row counts); default `source_mode: hf`; report missing sources and stop | `data/<task>/...`, data manifests |
| `02_measure_flops` | Run a trainer measurement pass for each method; count the first 100 micro-batches with `FlopCounterMode` | `outputs/<leaf>/flops/{vanilla,selfcond}/flops.json` |
| `03_train_vanilla` | Train VanillaFM with the formal config; no rate input from 02 | `outputs/<leaf>/vanilla/checkpoint-*/` |
| `04_train_selfcond` | Train SelfCondFM with the formal config; no rate input from 02 | `outputs/<leaf>/selfcond/checkpoint-*/` |
| `05_match_checkpoints` | `apply` stamps cumulative FLOPs in place (idempotent), then `audit` and SelfCond-anchored `match`; step back if needed; exit 2 when unmatched | updated `training_state.pt`, `audit.json`, `pairs.json` |
| `06_evaluate` | Score the selected pair; refuse an absent or unmatched `pairs.json` | per-instance evaluation results |
| `07_summarize` | Aggregate metrics, budgets, and provenance, including measured rates | `summary.json` |

Other leaves use these variants:

| Leaves | Stages | Dependency / FLOPs role |
| --- | --- | --- |
| `04_nfe_scaling/line_tracing` | 4: prepare -> locate checkpoints -> sweep NFE -> summarize | Requires 01's matched pair; no training or pair re-selection |
| `06_diffthinker_large_scale/task_specific/<task>` | 6: prepare -> measure FLOPs -> train selfcond -> evaluate -> score -> summarize | SelfCond cost is recorded; no training-FLOPs pairing against external baselines |
| `06_diffthinker_large_scale/unified` | same 6 stages | One joint VSP/Maze/TSP/Sudoku adapter, evaluated per task |

## run.sh, preflight, and profiles

`run.sh` first runs `scripts/common/preflight.sh` (offline config validation:
`experiment.id`, `save_steps == eval_every_steps`, method router, pairing
anchor), then the stages in order. It accepts `--profile paper|smoke`,
`--dry-run`, and `--stage NAME` (repeatable). Stage scripts are independently
callable and resolve `scripts/common/env.sh` from any depth.

`--profile smoke` needs its own small-budget config with
`experiment.mode: smoke`, supplied through `CONFIG_OVERRIDE`; the profile flag
does not shrink a paper config by itself.

## FLOPs measurement and pairing

- Stage 02 runs each method's trainer with `--measure_flops` and records the
  mean FLOPs of the first 100 micro-batches per method.
- Training (stages 03/04) does not consume the rates; skipping 02 still lets
  training run, and a missing rate surfaces at stage 05.
- Stage 05 stamps `cumulative_flops = rate * global_step * grad_accum *
  num_processes` into each checkpoint's `training_state.pt` in place, audits
  that none is missing, then selects the pair anchored at the paper SelfCond
  step: the highest SelfCond candidate whose nearest Vanilla satisfies
  `F_vanilla >= F_selfcond` (no gap threshold); accuracy plays no part in the
  selection.
- `scripts/common/flops.sh` is a manual convenience wrapper; no stage calls it.

The counting scope, validation rules, and selection algorithm are
maintained in [`flops_protocol.md`](flops_protocol.md); the implementation
lives in `src/experiment_tools/`, and stage scripts never re-implement it.

## Data and weights

The repo ships no dataset files and no model weights. Each leaf's
`01_prepare_data.sh` provisions and verifies its data before training
(sha256 + row counts); it stops with a report when a source is missing.
Base model weights are fetched by the prepare/train stages per each `config.yaml`.

Copy `.env.example` to `.env` and set:

| Var | Purpose |
| --- | --- |
| `HF_TOKEN` | not needed for dataset downloads; required for gated base models or uploads |
| `HF_DATASET_REPO` | optional full dataset id override |
| `HF_ENDPOINT` | HF endpoint; defaults to the mirror `https://hf-mirror.com` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | long-mult judge, long-text OCR |
| `QWEN3VL_CKPT` | Sudoku / Jigsaw local OCR |
| `DATA_SOURCE_MODE` | optional; `inherit_original` selects the local-checkout path |
| `ORIGINAL_REPO_ROOT` | `inherit_original`: original synthetic-data checkout (01 to 04) |
| `DIFFTHINKER_SOURCE` | EndoCoT/DiffThinker checkout for 05/06 (HF assembly when unset) |

The paper datasets, including line tracing and its NFE subset, download from
the [HF dataset repository](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data)
at a pinned revision (default `source_mode: hf`).
DiffThinker's upstream EndoCoT
images/GT are referenced from `internlm/EndoCoT-Data` + `InternLM/EndoCoT`.
Hosting, pins, and provenance: [`data.md`](data.md).

## Pretrained checkpoints

For reference, we have also released our trained checkpoints on
[Hugging Face](https://huggingface.co/alfalfa-w/self-conditioning-deep-reasoning).

## Quickstart: all 15 leaves

Commands start from the repository root. Prepare `.env`, install
`requirements.txt`, and provide CUDA GPUs and data access as above. 02/03 need
the API judge/OCR settings; Sudoku/Jigsaw need `QWEN3VL_CKPT`, vLLM, and
`qwen_vl_utils`; VSP needs `gymnasium`. Dry-run first with
`cd <leaf> && bash run.sh --dry-run` to inspect the config path and stage list.

| Command (from repo root) | Reproduces | Stages |
| --- | --- | --- |
| `cd scripts/01_line_tracing && bash run.sh --profile paper` | Line tracing m=50; from-scratch FLUX.2 `9M_LONG`; multiseed + examples | 7 |
| `cd scripts/02_long_multiplication && bash run.sh --profile paper` | Column multiplication; FLUX.2-klein LoRA; whole-worksheet API judge; multiseed | 7 |
| `cd scripts/03_long_text_rendering && bash run.sh --profile paper` | TinyStories; FLUX.2-klein LoRA + Qwen3 text; OCR/CER; optional multiseed extension | 7 |
| `cd scripts/04_nfe_scaling/line_tracing && bash run.sh --profile paper` | NFE 1 to 50 and inference-FLOPs axes over 01's selected pair; complete 01 first | 4 |
| `cd scripts/05_diffthinker_controlled/vsp && bash run.sh --profile paper` | Controlled VSP; FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7 |
| `cd scripts/05_diffthinker_controlled/maze && bash run.sh --profile paper` | Controlled Maze; FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7 |
| `cd scripts/05_diffthinker_controlled/tsp && bash run.sh --profile paper` | Controlled TSP; FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7 |
| `cd scripts/05_diffthinker_controlled/sudoku && bash run.sh --profile paper` | Controlled Sudoku; FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7 |
| `cd scripts/05_diffthinker_controlled/jigsaw && bash run.sh --profile paper` | Controlled Jigsaw; FLUX.2-klein LoRA; VanillaFM vs SelfCondFM | 7 |
| `cd scripts/06_diffthinker_large_scale/task_specific/vsp && bash run.sh --profile paper` | Large-scale VSP; Qwen-Image-Edit-2511 LoRA; recorded SelfCond cost | 6 |
| `cd scripts/06_diffthinker_large_scale/task_specific/maze && bash run.sh --profile paper` | Large-scale Maze; Qwen-Image-Edit-2511 LoRA; recorded SelfCond cost | 6 |
| `cd scripts/06_diffthinker_large_scale/task_specific/tsp && bash run.sh --profile paper` | Large-scale TSP; Qwen-Image-Edit-2511 LoRA; recorded SelfCond cost | 6 |
| `cd scripts/06_diffthinker_large_scale/task_specific/sudoku && bash run.sh --profile paper` | Large-scale Sudoku; Qwen-Image-Edit-2511 LoRA; recorded SelfCond cost | 6 |
| `cd scripts/06_diffthinker_large_scale/task_specific/jigsaw && bash run.sh --profile paper` | Large-scale Jigsaw; Qwen-Image-Edit-2511 LoRA; recorded SelfCond cost | 6 |
| `cd scripts/06_diffthinker_large_scale/unified && bash run.sh --profile paper` | One Qwen-Image adapter jointly trained on VSP/Maze/TSP/Sudoku; per-task evaluation | 6 |

The paper-label to config/checkpoint/artifact mapping is the index section of
[`reproduction.md`](reproduction.md#paper-figures-and-artifacts-index); validate
a fresh environment with its
[acceptance checklist](reproduction.md#acceptance-checks). Outputs land under
`outputs/<leaf>/`.

## Related documentation

- [`reproduction.md`](reproduction.md): environment and run contract,
  acceptance checks, figure/artifact index
- [`data.md`](data.md): datasets and provenance
- [`flops_protocol.md`](flops_protocol.md): measurement and pairing
- [`codebase.md`](codebase.md): directory guide
