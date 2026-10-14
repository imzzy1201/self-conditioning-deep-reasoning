#!/usr/bin/env python3
"""Report text files that contain a CR byte (i.e. CRLF line endings).

Why this exists: this repository targets Linux/Bash + CUDA, but its contents were
carried over from a Windows checkout. ``bash -n`` cannot detect a CRLF script —
it parses cleanly and even runs under Git Bash — yet a trailing CR silently
corrupts shell variables and paths (``x=1\\r`` != ``1``, so comparisons and ``cd``
misbehave), and on Linux a CRLF shebang is not executable
(``bad interpreter: ... bash\\r``). Only a byte-level inspection catches it.

Exempt: runtime ``data/**`` / ``outputs/**`` and ``results/reference/**``.
Inherited CSVs may use CRLF and must retain their recorded hashes; they are not
shell source files. The reference tree is a frozen copy of the historical
reference results and 25 of its manifests record the SHA256 and byte size *of the
CRLF bytes*; normalising it would break those recorded hashes. (Seven files had
cluster account paths in their provenance strings replaced with ``<CLUSTER_ROOT>``
on 2026-10-02, which changed their bytes but not their line endings.)

Usage:
    python scripts/common/check_line_endings.py [path ...]

With no arguments the whole repository is scanned. Exit status is 1 if any file
contains a CR byte, 0 otherwise. Intended as a structure-acceptance check
alongside ``bash -n``; see docs/reproduction.md.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}
# Frozen reference layer: CRLF bytes are part of the recorded contract.
SKIP_PREFIXES = ("results/reference/", "data/", "outputs/")
BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".pdf",
    ".ttf", ".otf", ".woff", ".woff2", ".safetensors", ".bin", ".pt",
    ".pth", ".ckpt", ".npz", ".npy", ".pyc", ".so", ".dll", ".dylib",
    ".gz", ".tgz", ".zip", ".tar", ".mp4", ".wav", ".mp3", ".parquet",
}


def _is_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return True
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(8192)
    except OSError:
        return True


def scan(paths):
    """Return (checked, offenders) where offenders are (relpath, cr_count)."""
    checked = 0
    offenders = []
    for target in paths:
        target = Path(target)
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        for p in files:
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if rel.startswith(SKIP_PREFIXES):
                continue
            if _is_binary(p):
                continue
            try:
                data = p.read_bytes()
            except OSError:
                continue
            checked += 1
            n = data.count(b"\r")
            if n:
                offenders.append((rel, n))
    return checked, offenders


def main(argv):
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0
    paths = [Path(a) for a in argv] or [ROOT]
    checked, offenders = scan(paths)
    if offenders:
        print(f"line-endings: {len(offenders)} of {checked} text files contain CR bytes:", file=sys.stderr)
        for rel, n in offenders:
            print(f"  {rel}  ({n} CR)", file=sys.stderr)
        print("line-endings: convert source files to LF (runtime data/outputs and frozen reference are exempt)", file=sys.stderr)
        return 1
    print(f"line-endings OK: {checked} text files checked, no CR bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
