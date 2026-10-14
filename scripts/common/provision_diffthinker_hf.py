#!/usr/bin/env python3
"""Assemble a DiffThinker/EndoCoT source tree from HuggingFace + upstream, for the
case where no local DIFFTHINKER_SOURCE checkout is provided.

Per decision D2 we host only OUR parts (the 12 derived `formatted/` CSVs, the
locally-generated `Sudoku/v2` and `Jigsaw`) under `diffthinker_local/` in our dataset
repo; the inherited images/GT come from upstream, referenced by pinned revision:
  - images: HF `internlm/EndoCoT-Data` @ 34c84daf…  (diffthinker_provenance.HF_SOURCE)
  - GT:     Git `InternLM/EndoCoT`      @ 8cf17c0…    (diffthinker_provenance.GIT_SOURCE)

It builds a staging tree with the three roots the existing
`provision_diffthinker_data.sh` expects (EndoCoT/, EndoCoT-Data/, formatted/) and
prints its path; that script then mirrors + pin-verifies + derives eval_unified.csv
unchanged. Nothing is fabricated: missing pieces raise.

Requires network to huggingface.co (or HF_ENDPOINT) + git. Not exercised offline.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pins import (DIFFTHINKER_LOCAL_SUBFOLDER, HF_DATASET_REVISION,
                          resolve_dataset_repo)  # noqa: E402
from diffthinker_provenance import GIT_SOURCE, HF_SOURCE  # noqa: E402
from hf_archives import materialize_archives, manifest_parts  # noqa: E402

DEFAULT_ENDPOINT = "https://hf-mirror.com"
# EndoCoT-Data sub-trees that come from upstream (ours overlay Sudoku/v2 + Jigsaw).
UPSTREAM_DATA_SUBDIRS = ("VSP", "Maze", "TSP", "Sudoku", "edit")


def _repo_id():
    return resolve_dataset_repo()


def _hf_url(repo):
    """Upstream HF repository URL -> dataset id (strip the .../datasets/ prefix)."""
    return repo.split("/datasets/")[-1]


def build_staging(dest=None):
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)
    from huggingface_hub import snapshot_download

    dest = Path(dest or Path(tempfile.mkdtemp(prefix="diffthinker_src_")))
    # Only a fully assembled tree may be reused after an interrupted download.
    marker = dest / ".hf_complete"
    revision = os.environ.get("HF_DATASET_REVISION") or HF_DATASET_REVISION
    identity = f"{_repo_id()}@{revision}"
    if marker.is_file() and marker.read_text() == identity:
        print(dest)
        return dest
    (dest / "EndoCoT-Data").mkdir(parents=True, exist_ok=True)
    (dest / "EndoCoT").mkdir(parents=True, exist_ok=True)

    # 1. Upstream images: internlm/EndoCoT-Data at the pinned revision.
    up_data = Path(snapshot_download(repo_id=_hf_url(HF_SOURCE["repository"]),
                                     repo_type="dataset", revision=HF_SOURCE["revision"]))
    up_data = materialize_archives(up_data)
    for sub in UPSTREAM_DATA_SUBDIRS:
        src = up_data / sub
        if not src.is_dir():
            raise FileNotFoundError(f"upstream EndoCoT-Data lacks '{sub}': {up_data}")
        target = dest / "EndoCoT-Data" / sub
        if sub == "Sudoku":
            # A real parent allows our v2 overlay without modifying the HF cache.
            target.mkdir(exist_ok=True)
            for entry in src.iterdir():
                if entry.name != "v2":
                    _link(entry, target / entry.name)
        else:
            _link(src, target)

    # 2. Upstream GT: InternLM/EndoCoT at the pinned commit (shallow clone).
    gt_url = GIT_SOURCE["repository"]
    gt_clone = dest / "_endocot_git"
    if not gt_clone.is_dir():
        subprocess.run(["git", "clone", "--quiet", gt_url, str(gt_clone)], check=True)
        subprocess.run(["git", "-C", str(gt_clone), "checkout", "--quiet", GIT_SOURCE["revision"]],
                       check=True)
    for gt in ("FrozenLake", "Maze", "TSP", "Sudoku"):
        src = gt_clone / gt
        if not src.is_dir():
            raise FileNotFoundError(f"EndoCoT GT lacks '{gt}': {gt_clone}")
        _link(src, dest / "EndoCoT" / gt)

    # 3. Our parts: diffthinker_local/{formatted, Sudoku_v2, Jigsaw} from our HF repo.
    ours = Path(snapshot_download(repo_id=_repo_id(), repo_type="dataset",
                                  revision=revision,
                                  allow_patterns=[f"{DIFFTHINKER_LOCAL_SUBFOLDER}/**", "MANIFEST.json"]))
    local = materialize_archives(ours / DIFFTHINKER_LOCAL_SUBFOLDER,
                                 manifest_parts(ours, DIFFTHINKER_LOCAL_SUBFOLDER))
    for name, target in (("formatted", dest / "formatted"),
                         ("Sudoku_v2", dest / "EndoCoT-Data" / "Sudoku" / "v2"),
                         ("Jigsaw", dest / "EndoCoT-Data" / "Jigsaw")):
        src = local / name
        if not src.is_dir():
            raise FileNotFoundError(f"our HF diffthinker_local lacks '{name}': {local}")
        target.parent.mkdir(parents=True, exist_ok=True)
        _link(src, target)

    marker.write_text(identity)
    print(dest)
    return dest


def _link(source, target):
    """Create an idempotent link; refuse to overwrite a different data source."""
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Data destination already contains other data: {target}")
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


if __name__ == "__main__":
    build_staging(sys.argv[1] if len(sys.argv) > 1 else None)
