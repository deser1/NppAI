import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_code_quality import evaluate


class TestCodeQualityEvaluator(unittest.TestCase):
    def test_pass_fail_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_solution.py").write_text(
                "import unittest\nfrom solution import f\n"
                "class T(unittest.TestCase):\n"
                " def test_value(self): self.assertEqual(f(), 42)\n", encoding="utf-8")
            manifest = [{"id": x, "test_file": "test_solution.py"}
                        for x in ("good", "bad", "missing", "invalid")]
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            submissions = root / "submissions"
            submissions.mkdir()
            (submissions / "good.py").write_text("def f(): return 42\n", encoding="utf-8")
            (submissions / "bad.py").write_text("def f(): return 0\n", encoding="utf-8")
            (submissions / "invalid.py").write_text("def f(:\n", encoding="utf-8")
            report = evaluate(root / "manifest.json", submissions)
            self.assertEqual(report["total"], 4)
            self.assertEqual(report["submitted"], 3)
            self.assertEqual(report["syntax_passed"], 2)
            self.assertEqual(report["tests_passed"], 1)
            self.assertEqual([x["status"] for x in report["cases"]],
                             ["passed", "test_failed", "missing", "syntax_failed"])


if __name__ == "__main__":
    unittest.main()
