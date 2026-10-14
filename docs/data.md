# Datasets

The paper datasets, including line tracing and its NFE subset, are hosted on
HuggingFace; `data/` is a git-ignored runtime directory. Downloads do not
require a token. Provisioning verifies metadata sha256 and row counts before
use and never rewrites or fabricates data.

- HF dataset repo: [alfalfa-w/self-conditioning-deep-reasoning-data](https://huggingface.co/datasets/alfalfa-w/self-conditioning-deep-reasoning-data). `HF_DATASET_REPO`
  overrides the full id; `HF_DATASET_NAMESPACE` is needed only when using a
  bare repo name.
- Download (the default): `data.source_mode: hf` runs
  `scripts/common/provision_hf_data.py`, which calls
  `snapshot_download(repo_id, revision=data.hf_revision,
  allow_patterns="<task>/**")`, honoring `HF_ENDPOINT` (defaults to the
  mirror). Independent tar parts are extracted into a reusable cache outside
  the immutable HF snapshot, then linked into the runtime layout. Set
  `HF_DATASET_EXTRACT_ROOT` to choose that cache.
- Upload (maintainers): `scripts/common/upload_data_to_hf.py`, run where
  `huggingface.co` is reachable; it prints the commit hash to pin as
  `data.hf_revision`.
- Pins: `scripts/common/dataset_pins.py` is the single source of truth for
  every sha256 and row count; the provisioners and the uploader all read it.

## Paper datasets and hosting status

| Dataset | HF path | splits (rows) | metadata sha256 | Notes |
| --- | --- | --- | --- | --- |
| Line tracing | `line_tracing/` | train 1,000,000, val 1,000 | `6f9f7145...`, `e8797c73...` | m=50, original packed shards |
| NFE eval | `line_tracing/nfe_eval/` | val 200 | `94b1c911...` | first 200 of the line-tracing val split |
| Long multiplication | `long_multiplication/` | data_200k 200,000, data_test100 100 | `033bafa5...`, `e5097457...` | PNG, variable size |
| Long text (TinyStories) | `tinystories/300k_no_edit_384x384/` | train 300,000, test 21,990 | `25c3944b...`, `0aa0eeb1...` | rendered 384x384; text is upstream |
| DiffThinker (ours) | `diffthinker_local/` | formatted/ (12 CSVs), Sudoku_v2 40,000, Jigsaw 30,000/300 | `dataset_pins.DIFFTHINKER_PINS` | locally derived CSVs plus locally generated Sudoku v2 / Jigsaw |

Long-multiplication archive names and internal directories use `data_200k` and
`data_test100`. Image bytes, ordered metadata bytes, and integrity pins are
unchanged. The local `inherit_original` adapter accepts the shared checkout's
historical split names. Frozen reference artifacts retain the paths recorded by
those runs. The three early development datasets (`a5_composite`,
`a5_composite_image`, `multiplication`) remain excluded.

## Regeneration references

Each synthetic generator package ships its own README with the generation
entry points and options; the `regenerate` source mode and the historical
`src/*/scripts/` entry scripts reference them:

| Dataset | Generator package |
| --- | --- |
| Line tracing | [`src/a5_line_trace_image_dataset/README.md`](../src/a5_line_trace_image_dataset/README.md) |
| Long multiplication | [`src/long_multiplication_dataset/README.md`](../src/long_multiplication_dataset/README.md) |
| Long text (TinyStories) | [`src/text_dataset/README.md`](../src/text_dataset/README.md) |

## Referenced upstream (not re-hosted)

| Data | Source | Revision |
| --- | --- | --- |
| EndoCoT-Data images (VSP/Maze/TSP/Sudoku-base/edit) | HF `internlm/EndoCoT-Data` | `34c84daf1403582804c0c1f778813a11e11ec6c9` |
| EndoCoT GT / eval trees | GitHub `InternLM/EndoCoT` | `8cf17c06a029aa50c91c0867cd3c547d2a91c658` |
| TinyStories text | HF `roneneldan/TinyStories` | not pinned upstream |
| COCO (Jigsaw source images) | upstream COCO | per DiffThinker `Jigsaw/gen.txt` recipe |

The DiffThinker mirror (`data/diffthinker/`) is assembled from these upstream
sources plus our `diffthinker_local/` by
`scripts/common/provision_diffthinker_data.sh`, then verified against the 12
pinned CSVs (row counts and sha256[:16]). Jigsaw test instances are locally
regenerated and differ from the official DiffThinker test set.

## DiffThinker provenance

This section is the provenance record for the experiments 05/06 data, based on
the paper appendix `app:diffthinker_setup`, the latest finished source runs'
training parameters, and the final checkpoint state.

### Image sources per task

| Task | Training images | Test images and answers | Current CSV rows (train/test) |
| --- | --- | --- | --- |
| VSP-Super | Inherited directly from the HF `internlm/EndoCoT-Data` release | Inherited from the `InternLM/EndoCoT/FrozenLake` Git tree; that project follows the DiffThinker setup | 37400 / 200 |
| Maze | Inherited directly from the same HF release | Inherited from `InternLM/EndoCoT/Maze` | 75000 / 300 |
| TSP | Inherited directly from the same HF release | Inherited from `InternLM/EndoCoT/TSP` | 30000 / 300 |
| Sudoku | The locally generated v2 set is used by the current controlled experiment, the latest large-model runs, and unified | Inherited from `InternLM/EndoCoT/Sudoku` | 40000 / 400 |
| Jigsaw | Generated locally from the COCO recipe in DiffThinker's `Jigsaw/gen.txt` | Regenerated locally; both images and answers differ from the official test instances | 30000 / 300 |

`formatted/` holds locally derived metadata; CSV conversion must not be
conflated with image generation. The VSP/Maze/TSP training CSVs are
three-column (`edit_image/image/prompt`) projections of the official
`metadata_edit_final.csv`; Jigsaw's is a reorder of the locally generated CSV;
`train_unified.csv` concatenates the four tasks (182,400 rows, no Jigsaw). The
test CSVs enumerate each task's GT directory, exactly 100 rows per difficulty.

The inherited versions are HF revision
`34c84daf1403582804c0c1f778813a11e11ec6c9` and EndoCoT Git commit
`8cf17c06a029aa50c91c0867cd3c547d2a91c658`. The HF download cache's
`.metadata` records the revision; the tracked trees of the four EndoCoT tasks
are unmodified. The EndoCoT README states that evaluation follows the
[DiffThinker](https://github.com/lcqysl/DiffThinker) setup.

### Sudoku v2: a UTC time error selects the wrong training set

The v2 `run_gen.sh` and `gen.log` live in the target of
`EndoCoT-Data/Sudoku/v2/`. On 2026-07-02, 16:35 to 17:26 Beijing time,
DiffThinker's `Sudoku/gen_image.py` generated 10,000 rows each for exist=30,
35, 40, 45 with max_workers=64, followed by the local `gen_csv_v2.py`.

The current `formatted/Sudoku/metadata_train.csv` was replaced with v2 at
2026-07-02 19:18:58 +0800. The latest finished run `<wandb-run-id>` has
`createdAt=2026-07-02T16:10:41Z`, which is 2026-07-03 00:10:41 +0800; training
ended around July 4, 04:15, and the final checkpoint landed at the same time.
The run started after the replacement, so it consumed the current v2 CSV
(sha256 first 16 hex: `743ec5d0ee0e2f5b`).
`task_specific/sudoku/config.yaml` points at that file.

The old June 26 run `<wandb-run-id>` had save/eval=2188 and last logged 21800
steps / 174400 samples, matching the old 35000-row revision; it is wrong to
claim both Sudoku runs consumed the 40000-row `.bak`. The 40000-row `.bak`
(`f0dba276d6483bda`) is kept as historical evidence only. Training seed=1201
does not prove the generation seed; the generation-time random state was not
saved.

### Jigsaw: the local directory is not the official Git data

`EndoCoT-Data/Jigsaw` is a symlink to DiffThinker/Jigsaw on a deprecated
mount, not a copy of the official Jigsaw release. In that clone, `git ls-tree
-r HEAD -- Jigsaw` shows upstream has only the test set, VisPuzzle, and code:
no training directory and no metadata_train.csv. The local training files are
untracked and dated 2026-05-21; "byte-identical to the current clone" does not
imply "inherited from the official release".

The local 30,000-row training quota is 4,000 each for 1x2, 1x3, 2x1, 3x1, 2x2
and 5,000 each for 3x3, 4x4, consistent with the COCO split_1 to split_7
recipe in the official `gen.txt`. The COCO source-image directory is
`COCO/{images,divide}` in the source tree. The training images are locally
constructed data from the upstream generation flow. The original image
mapping, generation seed, and full launch log have not been found; byte-exact
regeneration cannot be claimed.

Official and local test sets both hold 100 instances each for 2x2, 3x3, and
4x4, but all 600/600 PNGs, 298/300 answer TXTs, and 3 metadata.json files
differ. For example, the official answer for 2x2_00001 is `3 1 2 4` while the
local one is `2 3 4 1`. The generator also carries local modifications: the
font changed from Lato-Bold to Lato-Medium and the label y coordinate changed
to 0. These are locally regenerated instances, not merely a font re-render;
identical IDs prove neither identical source images nor identical
permutations. When the paper quotes external Jigsaw baselines, this test-set
difference must stay documented.

### Parameters and preparation-script corrections

The latest E06 runs use global batch=8, batch=1, accum=1, 8 processes,
linear schedule with warmup=1000, lr=1e-4, 5 epochs, workers=8, gradient
checkpointing, and find_unused_parameters. The six entries gained
`compile=true` and `log_every=100`. Jigsaw source-image sizes vary, but the
loader center-crops to 512x512 at divisor=16; formal training keeps batch=1.

The unified run `<wandb-run-id>` has eval_every_steps=0 and save_steps=11400.
The config restores periodic eval off; preflight allows eval=0, while a
positive eval interval still requires save==eval. The derived union CSV serves
manual diagnostics only; formal metrics come from `checkpoint-final`.

W&B `train/samples_seen` is the last logged point, not the final full count.
The final `training_state.pt` of VSP and Maze records 23375 and 46875 steps,
matching 37400 and 75000 rows at 5 epochs. TSP and Jigsaw end at 18750 steps,
matching 30000 rows at 5 epochs; some logs stop at 18700. A log deficit must
not be read as missing training files or CSV drift.

The prepare stage now emits per-task `content_provenance`, naming image and
metadata sources separately and verifying test image paths. Missing paths
return nonzero; broken symlinks in the mirror produce an error; the
old whole-root symlink's source is checked before migration. Source data stays
in the external tree; the prepare stage never regenerates missing images.

## Release flow (maintainers)

Keep configs and the DiffThinker default in `dataset_pins.HF_DATASET_REVISION`
pinned to the completed upload revision:

1. `python scripts/common/upload_data_to_hf.py --dry-run --exclude-line-tracing`:
   review the upload units and sizes.
2. `python scripts/common/upload_data_to_hf.py --exclude-line-tracing`: upload
   from a machine that can reach `huggingface.co` (set `HF_TOKEN` and the
   source roots); it prints the dataset commit hash. Existing archive releases
   use `--archive-staging <folder>` instead: the folder must contain the
   tar parts, `formatted/` metadata, `README.md`, and the checksum-indexed
   `MANIFEST.json`. The uploader verifies every archive and skips remote
   objects with the same SHA-256; expanded duplicates are rejected.
3. `python scripts/common/pin_hf_revision.py --revision <hash>`: write that
   hash into every `source_mode: hf` config's `data.hf_revision` (omit
   `--revision` to query HF for the current one). Only the `hf_revision:` line
   is rewritten; indentation and all other comments are preserved, and each
   config is re-parsed to confirm it is still valid YAML.
4. `python scripts/common/pin_hf_revision.py --check`: release gate; nonzero if
   any hf config is still `main`.
5. Commit the pinned configs.

## Other provisioning modes

- `inherit_original`: mirror from a local checkout (`ORIGINAL_REPO_ROOT` /
  `DIFFTHINKER_SOURCE`) with the same verification. For maintainers on the
  original cluster; select at runtime with `DATA_SOURCE_MODE=inherit_original`
  (no config edit needed).
- `regenerate`: re-run the generator under `src/*_dataset/`. It will not
  byte-match the frozen sets (historical generation seeds and fonts are not
  fully recovered), so it is for extension, not exact reproduction.

See [`reproduction.md`](reproduction.md) for the path and environment
contract.
