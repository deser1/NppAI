import json
import tempfile
import unittest
from pathlib import Path

from scripts.import_generation_outputs import extract_python, import_outputs


class TestGenerationOutputImport(unittest.TestCase):
    def test_raw_and_fenced(self):
        self.assertEqual(extract_python("def f(): pass"), "def f(): pass\n")
        fence = chr(96) * 3
        self.assertEqual(extract_python(fence + "python\ndef f(): pass\n" + fence),
                         "def f(): pass\n")

    def test_import_and_reject_stale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps([{"id": "task_a"}]), encoding="utf-8")
            (root / "outputs.json").write_text(
                json.dumps([{"id": "task_a", "output": "def f(): return 42"}]), encoding="utf-8")
            target = root / "submissions"
            self.assertEqual(import_outputs(root / "outputs.json", root / "manifest.json", target), 1)
            self.assertEqual((target / "task_a.py").read_text(encoding="utf-8"),
                             "def f(): return 42\n")
            with self.assertRaises(ValueError):
                import_outputs(root / "outputs.json", root / "manifest.json", target)

    def test_unknown_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.json").write_text(json.dumps([{"id": "task_a"}]), encoding="utf-8")
            for records in ([{"id": "other", "output": "pass"}],
                            [{"id": "task_a", "output": "pass"}] * 2):
                (root / "outputs.json").write_text(json.dumps(records), encoding="utf-8")
                with self.assertRaises(ValueError):
                    import_outputs(root / "outputs.json", root / "manifest.json", root / "out")


if __name__ == "__main__":
    unittest.main()
