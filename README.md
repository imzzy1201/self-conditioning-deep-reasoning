# Self-Conditioning Enables Deep Sequential Reasoning in Flow Matching

Reproduction of the paper's main experiments. Flow matching image models
(FLUX.2-klein and Qwen-Image) are trained on visual reasoning tasks, VanillaFM
and SelfCondFM checkpoints are paired at matched training FLOPs, and the
selected pairs are scored with evaluators specific to each task.

Each experiment is a leaf under `scripts/` that holds its own `config.yaml`,
`run.sh`, and numbered stage scripts. The repository guide is
[`docs/overview.md`](docs/overview.md).

## Repository structure

- `scripts/common/`: env resolution, config reader, preflight, data provisioners
- `scripts/01_line_tracing/`, `02_long_multiplication/`, `03_long_text_rendering/`:
  the three synthetic main tasks (seven stages each, plus multiseed and
  example scripts)
- `scripts/04_nfe_scaling/`: NFE 1 to 50 sweep over experiment 01's selected
  pair (four stages)
- `scripts/05_diffthinker_controlled/`: five controlled DiffThinker tasks on
  FLUX.2 LoRA, VanillaFM vs SelfCondFM (seven stages each)
- `scripts/06_diffthinker_large_scale/`: five Qwen-Image adapters, one per
  task, plus one unified adapter over four tasks (six stages each)
- `src/image_only/`: pure visual FLUX.2 training, sampling, diagnostics
- `src/image_text/`: flux2 / qwenimage / eval_external trainers and evaluators
- `src/experiment_tools/`: FLOPs measurement, pairing, eval summarization
- `src/analysis/`: plotting and tables
- `src/a5_line_trace_image_dataset/`, `src/long_multiplication_dataset/`,
  `src/text_dataset/`: dataset generators
- `docs/`: overview, reproduction guide, data, FLOPs protocol, codebase map
- `data/`, `outputs/`: runtime directories, ignored by git

## Environment setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Linux with CUDA GPUs; the training configs declare four processes (eight for
experiment 06). Fill in `.env`:

| Var | Purpose |
| --- | --- |
| `HF_TOKEN` | not needed for dataset downloads; required for gated base models or uploads |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | long multiplication judge, long text OCR |
| `QWEN3VL_CKPT` | Sudoku / Jigsaw local OCR (with vLLM + `qwen_vl_utils`) |
| `ORIGINAL_REPO_ROOT` / `DIFFTHINKER_SOURCE` | maintainer local checkouts |

VSP scoring also needs `gymnasium`.

## Data preparation

Every leaf's `01_prepare_data.sh` provisions and verifies its dataset (sha256
and row counts) before training:

- Line tracing, its NFE subset, long multiplication, and long text: downloaded
  from [alfalfa-w/self-conditioning-deep-reasoning-data](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data) at a pinned revision
  (default `source_mode: hf`).
- Original cluster data: use `DATA_SOURCE_MODE=inherit_original` and `ORIGINAL_REPO_ROOT`.
- DiffThinker tasks (05/06): mirrored from `DIFFTHINKER_SOURCE`, or assembled
  from pinned HF/upstream sources when it is unset.

Hosting, pins, and provenance: [`docs/data.md`](docs/data.md).

## Pretrained checkpoints

For reference, we have also released our trained checkpoints on
[Hugging Face](https://huggingface.co/alfalfa-w/self-conditioning-deep-reasoning).

## Running experiments

An example to run line tracing:

```bash
cd scripts/01_line_tracing
bash run.sh --dry-run            # offline config check; prints the resolved config and stage list
bash run.sh --profile paper      # stages 01 to 07
bash run.sh --stage 05_match_checkpoints --stage 06_evaluate   # selected stages only
```

Outputs land under `outputs/<leaf>/`. Each leaf README documents its
prerequisites, contracts, and outputs;
[`docs/overview.md`](docs/overview.md#quickstart-all-15-leaves) has the
quickstart table for all leaves.

## Documentation

- [`docs/overview.md`](docs/overview.md): repository guide (layout, experiment
  anatomy, quickstart)
- [`docs/reproduction.md`](docs/reproduction.md): environment, path/config
  contract, acceptance checks, paper figure and artifact index
- [`docs/data.md`](docs/data.md): datasets, pins, loading modes, DiffThinker
  provenance
- [`docs/flops_protocol.md`](docs/flops_protocol.md): FLOPs measurement and
  pairing
- [`docs/codebase.md`](docs/codebase.md): codebase map with one section per
  directory
