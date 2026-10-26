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

    def test_exact_model_bytes_are_bound_to_literal_legacy_diff_projection(self):
        import json,tempfile
        from pathlib import Path
        from scripts.portable_notepad_smoke import model_review_projection,bind_captured_proposal,capture_server
        code='def silnia(n):\n    return 1\n'
        display=model_review_projection(code)
        self.assertEqual(display,code+'\n')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);capture_server(root,root,8000)
            raw=root/'model-responses.jsonl'
            raw.write_text(json.dumps({'request':{'prompt':'real'},'response':{'output':code,'model_id':'own'}})+'\n')
            result=bind_captured_proposal({'code':display},raw)
            self.assertEqual(result['code'],code)
            self.assertEqual(result['review_code'],display)
            with self.assertRaises(ValueError):bind_captured_proposal({'code':display.replace('1','2')},raw)
