"""Offline checks for archive safety, multipart assembly and HF data contracts."""

import io
import hashlib
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_pins import resolve_dataset_repo
from hf_archives import materialize_archives, manifest_parts
import provision_diffthinker_hf


def write_tar(path, members):
    with tarfile.open(path, "w:") as archive:
        for name, value in members:
            info = tarfile.TarInfo(name)
            if value is None:
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/outside"
                archive.addfile(info)
            else:
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.environment = patch.dict(os.environ, {"HF_DATASET_EXTRACT_ROOT": str(self.root / "cache")})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_parts_plain_metadata_and_reuse(self):
        write_tar(self.snapshot / "train.part00.tar", [("train/images/one.png", b"image bytes")])
        write_tar(self.snapshot / "train.part01.tar", [("train/metadata.csv", b"ordered metadata\r\n")])
        plain = self.snapshot / "formatted"
        plain.mkdir()
        (plain / "metadata.csv").write_bytes(b"derived")
        output = materialize_archives(self.snapshot)
        self.assertEqual((output / "train/images/one.png").read_bytes(), b"image bytes")
        self.assertEqual((output / "train/metadata.csv").read_bytes(), b"ordered metadata\r\n")
        self.assertTrue((output / "formatted").is_symlink())
        self.assertEqual(materialize_archives(self.snapshot), output)
        self.assertFalse((self.snapshot / "train").exists())

    def test_unsafe_archives_never_publish_partial_output(self):
        for unsafe, value in [("../escape", b"bad"), ("/absolute", b"bad"), ("link", None)]:
            with self.subTest(unsafe=unsafe):
                write_tar(self.snapshot / "part.tar", [("safe/first", b"first"), (unsafe, value)])
                with self.assertRaises(ValueError):
                    materialize_archives(self.snapshot)
                self.assertFalse(list((self.root / "cache").glob("*/.complete.json")))
                self.assertFalse((self.root / "escape").exists())

    def test_duplicate_parts_fail(self):
        for name in ("part00.tar", "part01.tar"):
            write_tar(self.snapshot / name, [("train/image.png", b"same")])
        with self.assertRaises(FileExistsError):
            materialize_archives(self.snapshot)

    def test_missing_or_corrupt_part_fails_manifest_gate(self):
        archive = self.snapshot / "part00.tar"
        write_tar(archive, [("train/image.png", b"sample")])
        record = {"path": archive.name, "bytes": archive.stat().st_size,
                  "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        with self.assertRaises(ValueError):
            materialize_archives(self.snapshot, [record, {**record, "path": "part01.tar"}])
        with self.assertRaises(ValueError):
            materialize_archives(self.snapshot, [{**record, "sha256": "0" * 64}])
        self.assertFalse(list((self.root / "cache").glob("*/.complete.json")))

    def test_repository_override_precedes_config(self):
        with patch.dict(os.environ, {"HF_DATASET_REPO": "another/data"}):
            self.assertEqual(resolve_dataset_repo("alfalfa-w/self-conditioning-deep-reasoning-data"), "another/data")

    def test_nested_nfe_release_does_not_overlap_main_validation(self):
        line = self.snapshot / "line_tracing"
        subset = line / "nfe_eval"
        subset.mkdir(parents=True)
        write_tar(line / "val.part000.tar", [("val/metadata.csv", b"all validation")])
        write_tar(subset / "val.part000.tar", [("val/metadata.csv", b"first 200")])
        records = [{"path": str(path.relative_to(self.snapshot)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in self.snapshot.rglob("*.tar")]
        (self.snapshot / "MANIFEST.json").write_text(json.dumps({"archives": records}))
        main_parts = manifest_parts(self.snapshot, "line_tracing")
        subset_parts = manifest_parts(self.snapshot, "line_tracing/nfe_eval")
        self.assertEqual(len(main_parts), 1)
        self.assertEqual(len(subset_parts), 1)
        main = materialize_archives(line, main_parts)
        nfe = materialize_archives(subset, subset_parts)
        self.assertEqual((main / "val/metadata.csv").read_bytes(), b"all validation")
        self.assertEqual((nfe / "val/metadata.csv").read_bytes(), b"first 200")

    def test_diffthinker_assembly_does_not_modify_upstream_sudoku(self):
        upstream = self.root / "upstream"
        upstream.mkdir()
        members = [(f"{name}/example.png", b"upstream")
                   for name in provision_diffthinker_hf.UPSTREAM_DATA_SUBDIRS]
        write_tar(upstream / "images.tar", members)
        ours = self.root / "ours" / "diffthinker_local"
        ours.mkdir(parents=True)
        (ours / "formatted").mkdir()
        write_tar(ours / "Sudoku_v2.part00.tar", [("Sudoku_v2/30_train/image.png", b"ours")])
        write_tar(ours / "Jigsaw.part00.tar", [("Jigsaw/image.png", b"puzzle")])
        destination = self.root / "assembled"
        clone = destination / "_endocot_git"
        for name in ("FrozenLake", "Maze", "TSP", "Sudoku"):
            (clone / name).mkdir(parents=True)
        with patch("huggingface_hub.snapshot_download", side_effect=[str(upstream), str(ours.parent)]):
            result = provision_diffthinker_hf.build_staging(destination)
        self.assertEqual((result / "EndoCoT-Data/Sudoku/v2/30_train/image.png").read_bytes(), b"ours")
        self.assertFalse((materialize_archives(upstream) / "Sudoku/v2").exists())
        self.assertFalse((result / "EndoCoT-Data/Sudoku").is_symlink())
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected download")):
            self.assertEqual(provision_diffthinker_hf.build_staging(destination), result)


if __name__ == "__main__":
    unittest.main()
