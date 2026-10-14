import unittest

from scripts.diagnose_identifier_copy import identifier_span, summarize_case, summarize


class IdentifierDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.record = {"id": "probe", "function": "fn_a9z", "prompt": "Napisz funkcję fn_a9z(x).",
                       "answer": "def fn_a9z(x):\n    return x\n"}

    def test_next_token_offsets_use_bytes_after_unicode_prompt(self):
        sequence, start, end = identifier_span(self.record, 256)
        self.assertEqual(sequence[start + 1:end + 1], b"fn_a9z")
        self.assertEqual(sequence[start], ord(" "))
        self.assertEqual(sequence[end + 1], ord("("))

    def test_overflow_nul_and_wrong_reference_rejected(self):
        with self.assertRaises(ValueError):
            identifier_span(self.record, 10)
        for field, value in (("prompt", "bad\0prompt"), ("answer", "def wrong(x): pass")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                identifier_span({**self.record, field: value}, 256)

    def test_teacher_forcing_is_separate_from_autonomous_copying(self):
        item = summarize_case(self.record, list(b"fn_a9z"), "def wrong(x): pass", "def fn_a9z(x): broken(")
        self.assertTrue(item["teacher_forced_all_bytes_correct"])
        self.assertIsNone(item["teacher_forced_first_wrong_byte"])
        self.assertFalse(item["free_header_name_correct"])
        self.assertTrue(item["def_only_header_name_correct"])

    def test_partial_bytes_and_aggregation(self):
        item = summarize_case(self.record, list(b"fn_b9z"), " def fn_a9z(x): pass", "def wrong(x): pass")
        self.assertEqual(item["teacher_forced_first_wrong_byte"], 3)
        self.assertEqual(item["teacher_forced_correct_bytes"], 5)
        self.assertFalse(item["free_header_name_correct"])
        result = summarize([item, item])
        self.assertEqual(result["teacher_forced_correct_bytes"], 10)
        self.assertEqual(result["teacher_forced_total_bytes"], 12)
        self.assertEqual(result["teacher_forced_all_bytes_correct"], 0)
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            summarize_case(self.record, [], "", "")


if __name__ == "__main__":
    unittest.main()
