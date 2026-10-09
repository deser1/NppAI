import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.capture_model_code_outputs import capture


class TestCapture(unittest.TestCase):
    def test_capture_records_output_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine, model = root / "engine", root / "model.nppai"
            engine.write_bytes(b"binary")
            model.write_bytes(b"weights")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([{"id": "sum_positive", "prompt": "Write Python"}]),
                                encoding="utf-8")

            def fake_run(args, **kwargs):
                self.assertIn("[USER]: Write Python", args[2])
                Path(args[3]).write_text("def sum_positive(x): return 0", encoding="utf-8")
                return type("Result", (), {"returncode": 0, "stderr": ""})()

            with patch("scripts.capture_model_code_outputs.subprocess.run", side_effect=fake_run):
                outputs, metadata = capture(engine, model, manifest)
            self.assertEqual(outputs[0]["id"], "sum_positive")
            self.assertIn("def sum_positive", outputs[0]["output"])
            self.assertEqual(len(metadata["model_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
