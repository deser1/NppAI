import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    @patch("collect_real_model_candidates.sandbox_validate.validate")
    @patch("collect_real_model_candidates.sandbox_validate.load_cases", return_value=([{"function":"add","args":[],"kwargs":{},"expected":1}], "hash"))
    def test_failed_attempt_is_repaired_and_recorded(self, load, validate):
        validate.side_effect = [(1, "syntax error"), (0, "syntax passed"), (0, "tests passed")]
        router = SimpleNamespace(mode="local", generate=Mock(side_effect=[
            SimpleNamespace(output="bad code"), SimpleNamespace(output="def add(): return 1")]))
        rows = collect([{"task_id":"add","prompt":"Implement add","cases_file":"cases.json"}],
                       router, image="sha256:" + "a" * 64, max_attempts=3)
        self.assertEqual([r["attempt"] for r in rows], [1, 2])
        self.assertEqual(router.generate.call_count, 2)

    @patch("collect_real_model_candidates.sandbox_validate.validate", return_value=(2, "Docker unavailable"))
    @patch("collect_real_model_candidates.sandbox_validate.load_cases", return_value=([], "hash"))
    def test_infrastructure_failure_does_not_trigger_repair(self, load, validate):
        router = SimpleNamespace(mode="online", generate=Mock(return_value=SimpleNamespace(output="x")))
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            collect([{"task_id":"a","prompt":"x","cases_file":"cases.json"}],
                    router, image="sha256:" + "a" * 64, max_attempts=3)
        self.assertEqual(router.generate.call_count, 1)

    def test_repair_requires_docker_image(self):
        router = SimpleNamespace(mode="local", generate=Mock())
        with self.assertRaisesRegex(ValueError, "Docker image"):
            collect([], router, max_attempts=2)


if __name__ == "__main__":
    unittest.main()
