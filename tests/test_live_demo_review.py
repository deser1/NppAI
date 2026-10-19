import unittest
from scripts.portable_notepad_smoke import inspect_review_syntax


class LiveReviewTests(unittest.TestCase):
    def test_actual_diff_preserves_bad_model_output_without_execution(self):
        report='blocked\n--- a/smoke.py\n+++ b/smoke.py\n@@ -1,0 +1,2 @@\n+def silnia(n):\n+    f letab\n'
        result=inspect_review_syntax(report)
        self.assertEqual(result['code'],'def silnia(n):\n    f letab\n')
        self.assertFalse(result['syntax']['passed'])
        self.assertEqual(result['syntax']['execution'],'not_run')

    def test_status_text_is_not_a_generated_program(self):
        with self.assertRaises(ValueError):inspect_review_syntax('Ready; connection failed')
        result=inspect_review_syntax('--- a/x\n+++ b/x\n@@ -1,0 +1,1 @@\n+print("never executed")\n')
        self.assertTrue(result['syntax']['passed'])
        self.assertEqual(result['syntax']['functional_pass'],'not_measured')
