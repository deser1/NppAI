import unittest

from benchmarks.evaluate_rag_relevance import evaluate


class RelevanceMetricsTests(unittest.TestCase):
    def test_perfect_ranked_results(self):
        result = evaluate([{"query_id": "q1", "relevant_paths": ["a", "b"], "ranked_paths": ["a", "b"]}])
        self.assertEqual(result["metrics"]["1"]["recall"], 0.5)
        self.assertEqual(result["metrics"]["3"]["recall"], 1.0)
        self.assertEqual(result["metrics"]["3"]["mrr"], 1.0)
        self.assertAlmostEqual(result["metrics"]["3"]["ndcg"], 1.0)

    def test_missed_and_late_relevant_result(self):
        result = evaluate([{"query_id": "q1", "relevant_paths": ["a"], "ranked_paths": ["x", "a"]}])
        self.assertEqual(result["metrics"]["1"]["mrr"], 0.0)
        self.assertEqual(result["metrics"]["3"]["mrr"], 0.5)
        self.assertAlmostEqual(result["metrics"]["3"]["ndcg"], 1 / __import__("math").log2(3))

    def test_duplicate_sources_count_only_once(self):
        result = evaluate([{"query_id": "q1", "relevant_paths": ["a", "b"], "ranked_paths": ["a", "a", "b"]}])
        self.assertEqual(result["metrics"]["3"]["recall"], 1.0)
        self.assertAlmostEqual(result["metrics"]["3"]["ndcg"], 1.0)

    def test_empty_results(self):
        result = evaluate([{"query_id": "q1", "relevant_paths": ["a"], "ranked_paths": []}])
        self.assertEqual(result["metrics"]["5"]["mrr"], 0.0)

    def test_duplicate_query_ids_rejected(self):
        rows = [{"query_id": "q1", "relevant_paths": ["a"], "ranked_paths": []}] * 2
        with self.assertRaises(ValueError):
            evaluate(rows)

    def test_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([])


if __name__ == "__main__":
    unittest.main()
