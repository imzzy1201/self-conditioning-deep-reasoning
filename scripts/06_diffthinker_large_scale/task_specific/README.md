# 06_diffthinker_large_scale/task_specific: single-task SelfCond adapters (C07)

One SelfCond Qwen-Image-Edit-2511 LoRA adapter per task, trained 5 epochs to
`checkpoint-final/lora` and compared with published baselines. Each leaf is a
six-stage pipeline (`config.yaml`, `run.sh`, and stages 01 to
06). The differences between tasks collapse into `config.yaml`, so the stage
scripts are identical across the five leaves and each reads its own config via
`CFG()`.

| Task | id | Difficulties | Test set | Save/eval cadence | Score parser |
| --- | --- | --- | --- | --- | --- |
| `vsp/` | `diffthinker_large_vsp` | 16, 32 | 200 | 2338 | CPU FrozenLake (`gymnasium`) |
| `maze/` | `diffthinker_large_maze` | 8, 16, 32 | 300 | 4688 | CPU path-walk |
| `tsp/` | `diffthinker_large_tsp` | 12, 15, 18 | 300 | 1875 | CPU tour check (tolerance 0.5) |
| `sudoku/` | `diffthinker_large_sudoku` | 45, 40, 35, 30 | 400 | 2500 | vLLM + Qwen3-VL-8B (GPU) |
| `jigsaw/` | `diffthinker_large_jigsaw` | 2x2, 3x3, 4x4 | 300 | 1875 | vLLM + Qwen3-VL-8B (GPU) |

## Data

Data and GT roots live under `data/diffthinker/{formatted,EndoCoT,EndoCoT-Data}`
(see each leaf's `config.yaml` `data.*`). That tree is a symlink mirror of the
external EndoCoT/DiffThinker checkout named by `DIFFTHINKER_SOURCE` (set it in
`.env`). `scripts/common/provision_diffthinker_data.sh` creates and verifies
the mirror (pinned row counts and sha256); stage 01 calls it automatically and
stops with a report when the source is absent. Stage 01 only freezes paths and
row counts; it does not generate data.

All five tasks evaluate against DiffThinker's original per-task scorers via
`src/image_text/eval_external/eval_diffthinker.py` (stage 05), reading the
frozen final adapter's predictions from stage 04.

Jigsaw is not part of `tab:diffthinker_large`; its leaf feeds the separate
`tab:jigsaw_benchmark`. It is still trained here as a single-task adapter.

## How to run

```bash
cd scripts/06_diffthinker_large_scale/task_specific/vsp
bash run.sh --dry-run
bash run.sh --profile paper
```

Repeat for `maze`, `tsp`, `sudoku`, and `jigsaw`. After all leaves have written
their `by_difficulty.csv`, run the group aggregator:

```bash
cd scripts/06_diffthinker_large_scale
bash summarize.sh
```

Outputs land under `outputs/06_diffthinker_large_scale/<task>/`.

## Documentation

- [`docs/reproduction.md`](../../../docs/reproduction.md#paper-figures-and-artifacts-index):
  paper label, data, checkpoint, and frozen-artifact index
- [`docs/flops_protocol.md`](../../../docs/flops_protocol.md): FLOPs
  measurement protocol
- [`docs/data.md`](../../../docs/data.md#diffthinker-provenance):
  source tree and pinned metadata CSVs
- [`../README.md`](../README.md): group overview and contracts
