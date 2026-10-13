import unittest

from scripts.prompt_robustness import paired_outcomes, prompt_variants


class PromptRobustnessTests(unittest.TestCase):
    def test_variants_preserve_source_id_name_reference_and_task_text(self):
        record = {"id": "case", "function": "fn_a1b2c3", "prompt": "Implement fn_a1b2c3(x).",
                  "answer": "def fn_a1b2c3(x):\n    return x\n"}
        variants = prompt_variants(record)
        self.assertEqual(list(variants), ["original", "please_prefix", "python_prefix", "code_only_suffix"])
        self.assertEqual(variants["original"], record)
        self.assertEqual(record["prompt"], "Implement fn_a1b2c3(x).")
        for item in variants.values():
            for key in ("id", "function", "answer"):
                self.assertEqual(item[key], record[key])
            self.assertIn(record["prompt"], item["prompt"])
        self.assertEqual(variants["please_prefix"]["prompt"], "Please " + record["prompt"])
        self.assertEqual(variants["python_prefix"]["prompt"], "Write Python code. " + record["prompt"])
        self.assertEqual(variants["code_only_suffix"]["prompt"], record["prompt"] + " Return only Python code.")

    def test_paired_changes_are_per_case_not_a_difference_of_totals(self):
        original = [{"id": str(i), "staged": {"exact_response": x}} for i, x in enumerate((True, True, False, False))]
        changed = [{"id": str(i), "staged": {"exact_response": x}} for i, x in enumerate((True, False, True, False))]
        self.assertEqual(paired_outcomes(original, changed[::-1], "staged"),
                         {"retained_exact": 1, "lost_exact": 1, "gained_exact": 1, "neither_exact": 1})

    def test_missing_or_duplicate_ids_fail_instead_of_silently_pairing(self):
        case = {"id": "one", "staged": {"exact_response": True}}
        for original, changed in (([case], []), ([case, case], [case])):
            with self.assertRaises(ValueError):
                paired_outcomes(original, changed, "staged")


if __name__ == "__main__":
    unittest.main()
