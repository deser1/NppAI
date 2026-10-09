"""End-to-end smoke test: index a real temporary repository and score predictions."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


    def test_jsonl_escaped_fields_and_unicode_roundtrip(self):
        if len(sys.argv) < 2:
            self.skipTest("pass RagRepositoryPredictions executable as argument")
        executable = Path(sys.argv[1]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.cpp").write_text(
                "authentication token renderer processing\n", encoding="utf-8"
            )
            judgments = root / "judgments.jsonl"
            query_id = 'polski "test" \\ ' + chr(0x015B) + 'cie' + chr(0x017C) + 'ka'
            query = 'authentication "token" \\ renderer'
            judgments.write_text(
                json.dumps({"query_id": query_id, "query": query,
                            "relevant_sources": ["sample.cpp"]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [str(executable), str(root), str(judgments), "2"],
                capture_output=True, text=True, encoding="utf-8", timeout=30, check=True,
            )
            records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["query_id"], query_id)
            self.assertIsInstance(records[0]["ranked_sources"], list)

    def test_invalid_json_escape_is_rejected(self):
        if len(sys.argv) < 2:
            self.skipTest("pass RagRepositoryPredictions executable as argument")
        executable = Path(sys.argv[1]).resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.cpp").write_text("authentication token\n", encoding="utf-8")
            judgments = root / "judgments.jsonl"
            judgments.write_text(
                '{"query_id":"bad\\x","query":"authentication","relevant_sources":["sample.cpp"]}\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [str(executable), str(root), str(judgments), "2"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Invalid JSON string escape", result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
