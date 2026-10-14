import hashlib
import unittest

from scripts.evaluate_staged_generation import check_source, header_fits, score_output, valid_header


class StagedGenerationTests(unittest.TestCase):
    def test_generated_wrong_name_is_not_corrected_by_acceptance(self):
        header = "def wrong(x):\n"
        self.assertTrue(valid_header(header))
        self.assertTrue(header_fits("Implement requested(x).", header, 256))
        self.assertEqual(header, "def wrong(x):\n")

    def test_extra_code_inline_body_and_malformed_headers_rejected(self):
        for text in ("def f(x): pass\n", "def f(x):\n    return x\n", "def f(x):\ndef g(x):\n",
                     "def f(x):", "def f(:\n", "def f(x):\r\n", "import os\n"):
            with self.subTest(text=text):
                self.assertFalse(valid_header(text))
        self.assertFalse(header_fits("x" * 256, "def f(x):\n", 256))

    def test_complete_ast_name_and_exact_text_are_distinct(self):
        record = {"function": "f", "answer": "def f(x):\n    return x\n"}
        wrong_body = score_output(record, "def f(x):\n    return 0\n")
        self.assertTrue(wrong_body["requested_function"])
        self.assertFalse(wrong_body["reference_ast"])
        spacing = score_output(record, "def f(x):\n  return x\n")
        self.assertTrue(spacing["reference_ast"])
        self.assertFalse(spacing["exact_response"])
        incomplete = score_output(record, "def f(x):\n")
        self.assertFalse(incomplete["syntax_any_function"])

    def test_source_hash_manifest_and_preselected_configuration_are_verified(self):
        raw = b"checkpoint"
        manifest = {"frozen": "source"}
        report = {"model_sha256": hashlib.sha256(raw).hexdigest(), "source_manifest": manifest,
                  "steps": 6000, "seed": 42, "context": 256}
        check_source(report, raw, manifest, "source_manifest")
        check_source({**report, "steps": 10000}, raw, manifest, "source_manifest", expected_steps=10000)
        for field, value in (("model_sha256", "wrong"), ("source_manifest", {}), ("steps", 3000), ("context", 96)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_source({**report, field: value}, raw, manifest, "source_manifest")


if __name__ == "__main__":
    unittest.main()
