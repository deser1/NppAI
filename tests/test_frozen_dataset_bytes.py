"""Frozen manifest hashes must survive Git's Windows newline conversion policy."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FrozenDatasetBytesTests(unittest.TestCase):
    def test_all_frozen_split_bytes_match_their_manifests(self):
        for folder in sorted((ROOT / "evaluation").glob("instruction-v*")):
            manifest = json.loads((folder / "manifest.json").read_bytes())
            for split, metadata in manifest["splits"].items():
                with self.subTest(dataset=folder.name, split=split):
                    raw = (folder / (split + ".json")).read_bytes()
                    self.assertNotIn(b"\r\n", raw)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), metadata["sha256"])
                    self.assertEqual(len(json.loads(raw)), metadata["count"])

    @unittest.skipUnless(shutil.which("git"), "Git required for checkout regression")
    def test_autocrlf_checkout_preserves_frozen_bytes(self):
        relative = Path("evaluation/instruction-v3/validation.json")
        expected = (ROOT / relative).read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            repository, exported = Path(temporary) / "repository", Path(temporary) / "checkout"
            repository.mkdir()
            (repository / relative).parent.mkdir(parents=True)
            (repository / relative).write_bytes(expected)
            (repository / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
            def git(*args):
                return subprocess.run(["git", "-C", str(repository), *args],
                                      check=True, capture_output=True, timeout=15)
            git("init")
            git("config", "core.autocrlf", "true")
            git("add", ".gitattributes", relative.as_posix())
            git("checkout-index", "--all", "--prefix=" + exported.as_posix() + "/")
            self.assertEqual((exported / relative).read_bytes(), expected)
