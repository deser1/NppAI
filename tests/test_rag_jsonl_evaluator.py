import unittest

from scripts.evaluate_rag_jsonl import evaluate


class RagJsonlEvaluatorTests(unittest.TestCase):
    def test_perfect_ranking(self):
        labels = [{"query_id": "q1", "relevant_sources": ["a.cpp", "b.cpp"]}]
        ranked = [{"query_id": "q1", "ranked_sources": ["a.cpp", "b.cpp"]}]
        result = evaluate(labels, ranked, 3)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mrr_at_k"], 1.0)
        self.assertEqual(result["ndcg_at_k"], 1.0)

    def test_wrong_first_result(self):
        labels = [{"query_id": "q1", "relevant_sources": ["a.cpp"]}]
        ranked = [{"query_id": "q1", "ranked_sources": ["wrong.cpp", "a.cpp"]}]
        result = evaluate(labels, ranked, 3)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mrr_at_k"], 0.5)
        self.assertLess(result["ndcg_at_k"], 1.0)

    def test_missing_prediction_fails(self):
        with self.assertRaises(ValueError):
            evaluate([{"query_id": "q1", "relevant_sources": ["a.cpp"]}], [], 3)

    def test_invalid_k_fails(self):
        with self.assertRaises(ValueError):
            evaluate([], [], 0)


if __name__ == "__main__":
    unittest.main()
