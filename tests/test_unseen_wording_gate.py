"""Regression checks for unseen instruction wording and generated function names."""
import ast
import unittest

from scripts.build_instruction_dataset import build


def check_generated_code(source, requested_name):
    """Return syntax/name diagnostics; never execute model-generated code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return {"syntax": False, "requested_name": False}
    valid = len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
    return {
        "syntax": valid,
        "requested_name": valid and tree.body[0].name == requested_name,
    }


class UnseenWordingGateTests(unittest.TestCase):
    def test_held_out_prompts_have_no_training_leakage(self):
        splits = build()
        training = {r["prompt"] for r in splits["train"]}
        for split in ("validation", "test"):
            for record in splits[split]:
                self.assertNotIn(record["prompt"], training)

    def test_all_reference_answers_pass_syntax_and_name_gate(self):
        for split, records in build().items():
            for record in records:
                with self.subTest(split=split, function=record["function"]):
                    self.assertEqual(
                        check_generated_code(record["answer"], record["function"]),
                        {"syntax": True, "requested_name": True},
                    )

    def test_rejects_memorized_name_even_with_valid_code(self):
        result = check_generated_code("def fn_add_01(a, b):\n    return a + b\n", "fn_add_13")
        self.assertTrue(result["syntax"])
        self.assertFalse(result["requested_name"])

    def test_rejects_invalid_or_extra_code(self):
        self.assertFalse(check_generated_code("def broken(:", "broken")["syntax"])
        self.assertFalse(check_generated_code("def f():\n    pass\nprint('extra')", "f")["syntax"])

    def test_unseen_wording_variants_are_distinct(self):
        prompts = [r["prompt"] for r in build()["test"]]
        for original in prompts:
            variants = (
                "Could you implement this Python function? " + original,
                "Produce the Python implementation for the following task: " + original,
            )
            self.assertEqual(len(set((original,) + variants)), 3)


if __name__ == "__main__":
    unittest.main()
