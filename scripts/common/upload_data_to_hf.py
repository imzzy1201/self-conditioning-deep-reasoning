#!/usr/bin/env python3
"""Upload every dataset to the HuggingFace dataset repo (run where HF is reachable).

The repo hosts ONLY our self-generated / locally-derived data (decision D2); upstream
third-party data (EndoCoT / DiffThinker images + GT, TinyStories text, COCO) is
referenced by revision, never re-hosted. Layout + integrity pins come from
scripts/common/dataset_pins.py, so the upload and the `hf` provisioner agree.

  python scripts/common/upload_data_to_hf.py --dry-run         # list units + sizes
  python scripts/common/upload_data_to_hf.py                   # upload everything
  python scripts/common/upload_data_to_hf.py --only line_tracing

Env: HF_DATASET_NAMESPACE (else inferred from the token), HF_TOKEN, HF_ENDPOINT,
and the source roots ORIGINAL_REPO_ROOT / DIFFTHINKER_SOURCE.
After a successful upload it prints the commit hash to pin as `data.hf_revision`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_pins import (  # noqa: E402
    DIFFTHINKER_LOCAL_SUBFOLDER, DIFFTHINKER_PINS, HF_DATASET_REPO, HF_SUBFOLDER,
    PINS, SOURCE_ROOT, SOURCE_SPLITS, resolve_dataset_repo,
)
from verify_dataset import sha256_file  # noqa: E402

SYNTHETIC = ("line_tracing", "long_multiplication", "long_text_rendering")


def upload_archive_release(folder, dry_run=False):
    """Upload a checksum-indexed release; never mix tar and expanded layouts."""
    from huggingface_hub import HfApi, CommitOperationAdd
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    paths = [record["path"] for record in manifest["archives"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate archive paths in MANIFEST.json")
    for record in manifest["archives"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or any(p.startswith("new_") for p in relative.parts):
            raise ValueError(f"Invalid published archive name: {relative}")
        source = folder / relative
        if source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
            raise ValueError(f"Archive differs from MANIFEST.json: {relative}")
    metadata = [folder / "README.md", folder / "MANIFEST.json"]
    metadata += sorted((folder / DIFFTHINKER_LOCAL_SUBFOLDER / "formatted").rglob("*.csv"))
    metadata += sorted((folder / DIFFTHINKER_LOCAL_SUBFOLDER / "formatted").rglob("*.csv.bak"))
    if dry_run:
        print(f"[upload] verified {len(paths)} archive checksums; {len(metadata)} card/metadata files")
        print(f"[upload] DRY RUN -> {resolve_dataset_repo()}; no upload performed")
        return 0
    api = HfApi()
    rid = resolve_dataset_repo()
    info = api.repo_info(rid, repo_type="dataset", files_metadata=True)
    remote = {entry.rfilename: entry for entry in info.siblings}
    for record in manifest["archives"]:
        existing = remote.get(record["path"])
        if existing is not None and existing.lfs and existing.lfs.sha256 == record["sha256"]:
            continue
        print(f"[upload] {record['path']}", flush=True)
        api.create_commit(repo_id=rid, repo_type="dataset",
                          operations=[CommitOperationAdd(path_in_repo=record["path"],
                                                         path_or_fileobj=str(folder / record["path"]))],
                          commit_message=f"Upload paper archive {record['path']}")
    result = api.create_commit(repo_id=rid, repo_type="dataset",
                               operations=[CommitOperationAdd(path_in_repo=str(p.relative_to(folder)),
                                                              path_or_fileobj=str(p)) for p in metadata],
                               commit_message="Update paper data card, checksum manifest and metadata")
    print(f"[upload] completed dataset revision: {result.oid}")
    return 0


def _du(path):
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "?"


def _source_root(task):
    env_var, rel = SOURCE_ROOT[task]
    root = os.environ.get(env_var)
    if not root:
        raise SystemExit(f"[upload] set {env_var} to the checkout holding the original data for '{task}'")
    return Path(root), rel


def build_units(only=None, exclude_line_tracing=False):
    """Return a list of upload units: (label, resolver, path_in_repo, note).

    `resolver` is a callable(dry: bool) -> local folder to upload. For dry runs it
    only reports the source; for real runs it may stage a temp dir (caller cleans up).
    """
    units = []
    # --- synthetic: one unit per split (upload_folder source/<split> -> <sub>/<split>) ---
    for task in SYNTHETIC:
        if (only and not (task.startswith(only) or only.startswith(task + "/"))) or (exclude_line_tracing and task == "line_tracing"):
            continue
        root, rel = _source_root(task)
        base, sub = root / rel, HF_SUBFOLDER[task]
        for split in PINS[task]:
            src = base / SOURCE_SPLITS.get(task, {}).get(split, split)
            units.append((f"{task}/{split}", lambda d, s=src: s, f"{sub}/{split}", ""))
    # --- NFE: stage val/{metadata.csv (from metadata_first200.csv), shards/} ---
    def nfe_stage(dry):
        root, rel = _source_root("nfe_scaling")
        val = root / rel
        if dry:
            return val
        tmp = Path(tempfile.mkdtemp(prefix="nfe_"))
        (tmp / "val").mkdir()
        shutil.copy2(val / "metadata_first200.csv", tmp / "val" / "metadata.csv")
        shutil.copytree(val / "shards", tmp / "val" / "shards", symlinks=False)
        return tmp / "val"
    if not exclude_line_tracing and (not only or "nfe_scaling".startswith(only)):
        units.append(("nfe_scaling/val", nfe_stage, f"{HF_SUBFOLDER['nfe_scaling']}/val",
                      "staged: metadata_first200.csv->metadata.csv + shared val shards"))
    # --- diffthinker_local: only OUR parts (formatted CSVs + Sudoku/v2 + Jigsaw) ---
    dt = os.environ.get("DIFFTHINKER_SOURCE") if not only or "diffthinker".startswith(only) or only.startswith("diffthinker/") else None
    if dt:
        dt = Path(dt)
        def formatted_stage(dry):
            if dry:
                return dt / "formatted"
            tmp = Path(tempfile.mkdtemp(prefix="diffthinker_csv_"))
            for relative in DIFFTHINKER_PINS:
                destination = tmp / Path(relative).relative_to("formatted")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dt / relative, destination)
            return tmp
        for label, src, target in (
            ("diffthinker/formatted", dt / "formatted", f"{DIFFTHINKER_LOCAL_SUBFOLDER}/formatted"),
            ("diffthinker/sudoku_v2", dt / "EndoCoT-Data/Sudoku/v2", f"{DIFFTHINKER_LOCAL_SUBFOLDER}/Sudoku_v2"),
            ("diffthinker/jigsaw", dt / "EndoCoT-Data/Jigsaw", f"{DIFFTHINKER_LOCAL_SUBFOLDER}/Jigsaw"),
        ):
            resolver = formatted_stage if label.endswith("/formatted") else lambda d, s=src: s
            units.append((label, resolver, target, "staged pinned CSVs" if label.endswith("/formatted")
                          else "ours only; upstream images referenced"))
    elif not only or "diffthinker".startswith(only) or only.startswith("diffthinker/"):
        units.append(("diffthinker_local", lambda d: None, DIFFTHINKER_LOCAL_SUBFOLDER,
                      "SKIPPED: set DIFFTHINKER_SOURCE to include formatted/ + Sudoku/v2 + Jigsaw"))
    return [u for u in units if not only or u[0].startswith(only)]


def repo_id():
    return resolve_dataset_repo()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list units + sizes, upload nothing")
    ap.add_argument("--only", help="only units whose label starts with this prefix")
    ap.add_argument("--no-size", action="store_true", help="skip du sizing (faster dry-run)")
    ap.add_argument("--exclude-line-tracing", action="store_true",
                    help="exclude line-tracing and its NFE subset")
    ap.add_argument("--archive-staging", help="upload a prepared archive release with MANIFEST.json")
    args = ap.parse_args()
    if args.archive_staging:
        return upload_archive_release(args.archive_staging, args.dry_run)

    units = build_units(args.only, args.exclude_line_tracing)
    rid = None if args.dry_run else repo_id()

    if args.dry_run:
        print(f"[upload] DRY RUN -> dataset repo '{HF_DATASET_REPO}' "
              f"(namespace {os.environ.get('HF_DATASET_NAMESPACE') or '<from token>'})")
        for label, resolver, target, note in units:
            src = resolver(True)
            size = "" if (args.no_size or src is None or not Path(src).exists()) else f"  [{_du(src)}]"
            exists = "" if src is not None and Path(src).exists() else "  (MISSING)"
            print(f"  {label:28} -> {target:34}{size}{exists}  {note}")
        print("[upload] no upload performed. Re-run without --dry-run where huggingface.co is reachable.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(rid, repo_type="dataset", exist_ok=True)
    remote_files = api.list_repo_files(rid, repo_type="dataset")
    # This CLI uploads expanded folders. Existing archive releases must be
    # replaced deliberately rather than mixing two representations of a split.
    for label, resolver, target, note in units:
        if any(path.startswith(target + ".part") and path.endswith(".tar") for path in remote_files):
            raise SystemExit(f"[upload] {target} is already an archive release. "
                             "Replace its tar parts in an atomic HF commit; "
                             "do not upload an expanded duplicate beside them.")
    staged = []
    try:
        for label, resolver, target, note in units:
            src = resolver(False)
            if src is None:
                print(f"[upload] SKIP {label}: {note}")
                continue
            src = Path(src)
            if "staged pinned CSVs" in note:
                staged.append(src)
            if not src.exists():
                raise SystemExit(f"[upload] missing source for {label}: {src}")
            print(f"[upload] {label}: {src} -> {rid}::{target}", flush=True)
            api.upload_folder(repo_id=rid, repo_type="dataset", folder_path=str(src),
                              path_in_repo=target)
            if "staged" in note and src.parent.name.startswith("nfe_"):
                staged.append(src.parent)
        info = api.repo_info(rid, repo_type="dataset")
        print(f"\n[upload] done. Dataset revision: {info.sha}")
        print("[upload] Now pin it into every hf config's data.hf_revision:")
        print(f"         python scripts/common/pin_hf_revision.py --revision {info.sha}")
        print("         (or omit --revision to query HF; then pin_hf_revision.py --check as a gate)")
        return 0
    finally:
        for d in staged:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
