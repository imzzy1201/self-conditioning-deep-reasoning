"""Shared dataset integrity gates: sha256 + logical row count + first image/shard ref.

Factored out of provision_original_data.py so the `inherit_original`, `hf`, and NFE
provisioners apply identical checks. sha256 is compared by the pin's length, so both
full 64-hex pins (synthetic) and 16-hex prefixes (DiffThinker CSVs) work. These gates
never fabricate data — any mismatch raises.
"""

import csv
import hashlib
from pathlib import Path


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Streaming sha256 of a file (handles multi-GB metadata without loading it)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split(metadata, expected_rows=None, expected_sha=None, label="",
                 check_first_image=True):
    """Verify one split's metadata.csv. Returns {rows, metadata_sha256, source}.

    - sha256 of the CSV must match `expected_sha` (compared over the pin's length, so
      a 16-hex prefix pin also validates). CSV hashes pin ordering + prompts + the
      image references, not just the row count.
    - logical data-row count (excl. header) must equal `expected_rows`.
    - when `check_first_image`, the first row's `image`/`edit_image` must resolve: a
      `shard:`-prefixed value requires `shards/shard_index.json`; a path value must
      exist relative to the metadata dir.
    """
    metadata = Path(metadata)
    if not metadata.is_file():
        raise FileNotFoundError(f"{label}missing metadata csv: {metadata}")
    digest = sha256_file(metadata)
    if expected_sha and digest[:len(expected_sha)] != expected_sha:
        raise ValueError(f"{label}metadata sha256 {digest} != pinned {expected_sha}: {metadata}")
    count = 0
    first = None
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            first = first or row
            count += 1
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"{label}{metadata}: {count} rows, expected {expected_rows}")
    if check_first_image:
        for key in ("image", "edit_image"):
            value = (first or {}).get(key, "")
            if value.startswith("shard:"):
                if not (metadata.parent / "shards" / "shard_index.json").is_file():
                    raise FileNotFoundError(f"{label}missing shard index: {metadata.parent}")
            elif value and not (metadata.parent / value).is_file():
                raise FileNotFoundError(f"{label}missing image: {metadata.parent / value}")
    return {"rows": count, "metadata_sha256": digest, "source": str(metadata.parent.resolve())}
