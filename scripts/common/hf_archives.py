"""Safely materialize independent tar parts outside the immutable HF snapshot."""

import fcntl
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def extract_parts(archives, destination):
    """Extract regular files/directories; reject links, escapes and duplicate files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for path in archives:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in member.name:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if not member.isdir() and not member.isfile():
                    raise ValueError(f"Unsupported archive member: {member.name}")
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output, archive.extractfile(member) as source:
                        shutil.copyfileobj(source, output, 1024 * 1024)


def materialize_archives(snapshot_root, expected_archives=None):
    """Reuse a completed extraction; interrupted extractions never become visible.

    Archive parts are independent tar files, extracted into one shared root. Plain
    directories (e.g. DiffThinker's formatted CSVs) are linked from the snapshot.
    HF LFS object identities and sizes key the cache; downloads remain untouched.
    """
    snapshot_root = Path(snapshot_root)
    # A manifest describes one release directory. Nested releases (such as the
    # line-tracing NFE subset) are downloaded with its parent but extracted alone.
    archives = sorted(snapshot_root.glob("*.tar") if expected_archives is not None
                      else snapshot_root.rglob("*.tar"))
    if expected_archives is not None:
        actual = {str(p.relative_to(snapshot_root)): p for p in archives}
        expected = {record["path"]: record for record in expected_archives}
        if set(actual) != set(expected):
            raise ValueError(f"Archive parts differ from manifest: missing={sorted(set(expected) - set(actual))}, "
                             f"unexpected={sorted(set(actual) - set(expected))}")
        for name, path in actual.items():
            if path.stat().st_size != expected[name]["bytes"]:
                raise ValueError(f"Archive size differs from manifest: {name}")
    if not archives:
        return snapshot_root
    identities = [(str(p.relative_to(snapshot_root)), str(p.resolve()), p.stat().st_size)
                  for p in archives]
    key = hashlib.sha256(json.dumps([str(snapshot_root), identities, expected_archives]).encode()).hexdigest()
    base = Path(os.environ.get("HF_DATASET_EXTRACT_ROOT",
                               str(Path.home() / ".cache" / "image-reasoning-data")))
    base.mkdir(parents=True, exist_ok=True)
    target = base / key
    with (base / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / ".complete.json").is_file():
            return target
        temporary = Path(tempfile.mkdtemp(prefix=f"{key}.", dir=base))
        try:
            if expected_archives is not None:
                for name, path in actual.items():
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != expected[name]["sha256"]:
                        raise ValueError(f"Archive sha256 differs from manifest: {name}")
            extract_parts(archives, temporary)
            for entry in snapshot_root.iterdir():
                if entry.is_dir() and not list(entry.rglob("*.tar")):
                    dest = temporary / entry.name
                    if dest.exists():
                        raise FileExistsError(f"Archive and plain directory overlap: {entry}")
                    dest.symlink_to(entry.resolve(), target_is_directory=True)
            (temporary / ".complete.json").write_text(json.dumps(identities) + "\n")
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return target


def manifest_parts(snapshot, subfolder):
    """Read the published archive index; upstream releases can omit this file."""
    path = Path(snapshot) / "MANIFEST.json"
    if not path.is_file():
        return None
    prefix = subfolder.rstrip("/") + "/"
    records = json.loads(path.read_text())["archives"]
    return [{**record, "path": record["path"][len(prefix):]} for record in records
            if record["path"].startswith(prefix)
            and "/" not in record["path"][len(prefix):]]
