import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from collect_real_model_candidates import collect


class CollectorTests(unittest.TestCase):
    def test_records_real_response(self):
        router = SimpleNamespace(mode="local", generate=Mock(return_value=SimpleNamespace(output="def add(a,b): return a+b")))
        result = collect([{"task_id":"add","prompt":"Implement add","cases_file":"cases.json"}], router)
        self.assertEqual(result[0]["source"], "def add(a,b): return a+b")
        self.assertEqual(result[0]["backend"], "local")
        self.assertEqual(result[0]["attempt"], 1)
        self.assertEqual(router.generate.call_count, 1)

    def test_rejects_duplicate_task(self):
        task = {"task_id":"a","prompt":"x","cases_file":"cases.json"}
        router = SimpleNamespace(mode="local", generate=Mock(return_value=SimpleNamespace(output="x")))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            collect([task, task], router)

    def test_rejects_hybrid_provenance(self):
        router = SimpleNamespace(mode="hybrid", generate=Mock(return_value=SimpleNamespace(output="x")))
        with self.assertRaisesRegex(ValueError, "provenance"):
            collect([{"task_id":"a","prompt":"x","cases_file":"cases.json"}], router)


if __name__ == "__main__":
    unittest.main()
