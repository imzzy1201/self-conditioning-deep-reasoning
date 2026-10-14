#!/usr/bin/env python3
"""Pin `data.hf_revision` in every hf-mode config to the uploaded dataset revision.

Flow after uploading the data (see scripts/common/upload_data_to_hf.py, which prints
the dataset commit hash):

    python scripts/common/pin_hf_revision.py --revision <commit-hash>
    # or let it query HuggingFace for the repo's current revision:
    HF_DATASET_NAMESPACE=<org> python scripts/common/pin_hf_revision.py

Only the `hf_revision:` line of each config whose `data.source_mode` is `hf` is
rewritten (indentation preserved, comment normalized); nothing else in the file is
touched, so all other comments/formatting survive. Every rewritten config is
re-parsed to confirm it is still valid YAML.

  --dry-run   show the changes without writing
  --check     exit non-zero if any hf config is still unpinned (revision 'main');
              use this as a release gate after pinning
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from dataset_pins import resolve_dataset_repo  # noqa: E402

DEFAULT_ENDPOINT = "https://hf-mirror.com"
UNPINNED = {"", "main", "null", "none"}
REV_RE = re.compile(r"^[0-9a-f]{7,40}$")
LINE_RE = re.compile(r"^([ \t]*)hf_revision:[ \t]*\S+.*$", re.M)


def resolve_repo_id():
    return resolve_dataset_repo()


def fetch_revision(repo_id):
    os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("[pin] huggingface_hub not installed; pass --revision explicitly.")
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT)
    print(f"[pin] querying {repo_id} @ {endpoint} for the current revision...", flush=True)
    return HfApi().repo_info(repo_id, repo_type="dataset").sha


def hf_configs():
    files = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "*config.yaml"], text=True).split()
    out = []
    for rel in files:
        p = REPO / rel
        t = p.read_text(encoding="utf-8")
        if re.search(r"^[ \t]*source_mode:[ \t]*hf\b", t, re.M):
            out.append(p)
    return out


def current_revision(text):
    m = re.search(r"^[ \t]*hf_revision:[ \t]*(\S+)", text, re.M)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", help="dataset commit hash; if omitted, query HuggingFace")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if any hf config is still unpinned ('main')")
    args = ap.parse_args()

    cfgs = hf_configs()
    if not cfgs:
        raise SystemExit("[pin] no configs with data.source_mode: hf found")

    if args.check:
        unpinned = [str(c.relative_to(REPO)) for c in cfgs
                    if (current_revision(c.read_text(encoding="utf-8")) or "").lower() in UNPINNED]
        from dataset_pins import HF_DATASET_REVISION
        if HF_DATASET_REVISION.lower() in UNPINNED:
            unpinned.append("scripts/common/dataset_pins.py (DiffThinker default)")
        if unpinned:
            print(f"[pin] CHECK FAILED — {len(unpinned)} hf config(s) still unpinned (hf_revision: main):")
            for u in unpinned:
                print("   ", u)
            return 1
        print(f"[pin] CHECK OK — all {len(cfgs)} hf configs pinned.")
        return 0

    rev = args.revision or fetch_revision(resolve_repo_id())
    rev = (rev or "").strip()
    if not REV_RE.fullmatch(rev):
        raise SystemExit(f"[pin] revision {rev!r} doesn't look like a git sha (7-40 hex). "
                         f"Pass --revision <hash> from the upload output.")

    changed = 0
    for p in cfgs:
        t = p.read_text(encoding="utf-8")
        new = LINE_RE.sub(lambda m: f"{m.group(1)}hf_revision: {rev}"
                                    f"{' ' * max(1, 21 - len(rev))}# pinned to the uploaded HF dataset revision",
                          t)
        rel = p.relative_to(REPO)
        if new != t:
            if not args.dry_run:
                p.write_text(new, encoding="utf-8")
                yaml.safe_load(p.read_text(encoding="utf-8"))  # fail loudly if we broke YAML
            changed += 1
            print(f"[pin]{' (dry-run)' if args.dry_run else ''} {rel}: hf_revision -> {rev}")
        else:
            print(f"[pin] {rel}: already {rev}")
    print(f"[pin] {'would pin' if args.dry_run else 'pinned'} {changed}/{len(cfgs)} config(s) to {rev}")
    pins = HERE / "dataset_pins.py"
    pins_text = pins.read_text()
    new_pins = re.sub(r'^HF_DATASET_REVISION = .*$',
                      f'HF_DATASET_REVISION = "{rev}"  # pinned uploaded dataset revision',
                      pins_text, flags=re.M)
    if new_pins != pins_text:
        if not args.dry_run:
            pins.write_text(new_pins)
        print(f"[pin]{' (dry-run)' if args.dry_run else ''} DiffThinker default -> {rev}")
    if not args.dry_run:
        print("[pin] done. Commit the configs; reproduction now downloads this exact revision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
