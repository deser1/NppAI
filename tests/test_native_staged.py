"""Native subprocess capture preserves bytes, limits and failure evidence."""
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from scripts.evaluate_native_staged import native_generate, summarize
from scripts.evaluate_staged_generation import score_output


class NativeStagedTests(unittest.TestCase):
    def test_prefix_and_output_bytes_and_budget_are_preserved(self):
        prefix = "def f(a):\n"
        def runner(command, **kwargs):
            self.assertEqual(command[-1], "160")
            self.assertEqual(command[2], "[USER]: task\n[AI]:\n")
            self.assertEqual(Path(command[4]).read_bytes(), prefix.encode())
            self.assertEqual(kwargs["errors"], "replace")
            Path(command[3]).write_bytes(b"def f(a):\r\n    return a\r\n")
            return subprocess.CompletedProcess(command, 0, "ok", "")
        with patch("scripts.evaluate_native_staged.subprocess.run", side_effect=runner):
            output, diagnostic = native_generate("runner", "model", "task", prefix)
        self.assertIn("\r\n", output)
        self.assertFalse(diagnostic["infrastructure_error"])

    def test_empty_generation_is_distinct_from_runner_failure(self):
        for code, infrastructure_error in ((5, False), (1, True), (0, True)):
            with patch("scripts.evaluate_native_staged.subprocess.run",
                       return_value=subprocess.CompletedProcess([], code, "", "error")):
                output, diagnostic = native_generate("runner", "model", "task")
            self.assertEqual(output, "")
            self.assertEqual(diagnostic["infrastructure_error"], infrastructure_error)
            self.assertEqual(diagnostic["returncode"], code)

    def test_timeout_is_recorded_as_infrastructure_failure(self):
        with patch("scripts.evaluate_native_staged.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("runner", 120)):
            output, diagnostic = native_generate("runner", "model", "task")
        self.assertEqual(output, "")
        self.assertTrue(diagnostic["infrastructure_error"])
        self.assertEqual(diagnostic["timeout"], 120)

    def test_correct_syntax_wrong_name_does_not_become_exact_success(self):
        record = {"function": "right", "answer": "def right(a):\n    return a\n"}
        metrics = score_output(record, "def wrong(a):\n    return a\n")
        case = {"header_accepted": True, "header_exact_reference": False,
                "staged": metrics, "single_model": metrics}
        summary = summarize([case])
        self.assertEqual(summary["staged"]["syntax_any_function"], 1)
        self.assertEqual(summary["staged"]["requested_function"], 0)
        self.assertEqual(summary["staged"]["exact_response"], 0)
