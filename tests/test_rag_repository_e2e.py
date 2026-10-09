"""End-to-end smoke test: index a real temporary repository and score predictions."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_rag_jsonl import evaluate, load_jsonl


class RepositoryRagEndToEnd(unittest.TestCase):
    def test_index_predict_and_score(self):
        if len(sys.argv) < 2:
            self.skipTest("pass RagRepositoryPredictions executable as argument")
        executable = Path(sys.argv[1]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "auth.cpp").write_text(
                "validate bearer authentication token before request dispatch\n", encoding="utf-8"
            )
            (root / "render.cpp").write_text(
                "render directx texture shader vertex buffer\n", encoding="utf-8"
            )
            judgments = root / "judgments.jsonl"
            judgments.write_text(
                json.dumps({"query_id": "auth", "query": "bearer authentication token",
                            "relevant_sources": ["auth.cpp"]}) + "\n", encoding="utf-8"
            )
            result = subprocess.run(
                [str(executable), str(root), str(judgments), "2"],
                capture_output=True, text=True, timeout=30, check=True
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(result.stdout, encoding="utf-8")
            scores = evaluate(load_jsonl(judgments), load_jsonl(predictions), 2)
            self.assertEqual(scores["queries"], 1)
            self.assertEqual(scores["recall_at_k"], 1.0)
            self.assertEqual(scores["mrr_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
