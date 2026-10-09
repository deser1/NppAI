import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_model_code_evaluation import run_pipeline


class TestModelCodeEvaluation(unittest.TestCase):
    def test_end_to_end_with_mock_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"id": "sample", "prompt": "Implement f",
                                              "test_file": "test_sample.py"}]), encoding="utf-8")
            (root / "test_sample.py").write_text(
                "import unittest\nfrom solution import f\n"
                "class TestSolution(unittest.TestCase):\n"
                "    def test_f(self): self.assertEqual(f(), 42)\n", encoding="utf-8")
            with patch("scripts.run_model_code_evaluation.capture",
                       return_value=([{"id": "sample", "output": "def f(): return 42"}],
                                     {"model_sha256": "abc"})):
                result = run_pipeline("engine", "model", manifest, root / "results")
            self.assertEqual(result["tests_passed"], 1)
            self.assertTrue((root / "results" / "code-quality-results.json").is_file())
            self.assertTrue((root / "results" / "generation-metadata.json").is_file())

    def test_existing_results_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            root.joinpath("previous.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                run_pipeline("engine", "model", "manifest", root)


if __name__ == "__main__":
    unittest.main()
