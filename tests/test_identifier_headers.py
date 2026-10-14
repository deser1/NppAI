import unittest

from scripts.train_identifier_headers import header_records, score_header


class IdentifierHeadersTests(unittest.TestCase):
    def setUp(self):
        self.record = {"id": "first", "function": "fn_a1b2c3", "prompt": "Napisz fn_a1b2c3(x).",
                       "answer": "def fn_a1b2c3(x):\n    return x\n"}

    def test_only_reference_response_changes_and_styles_are_retained(self):
        second = {**self.record, "id": "style2", "prompt": "Return x using fn_a1b2c3(x)."}
        result = header_records([self.record, second])
        self.assertEqual([r["id"] for r in result], ["first", "style2"])
        self.assertEqual([r["prompt"] for r in result], [self.record["prompt"], second["prompt"]])
        self.assertEqual(result[0]["answer"], "def fn_a1b2c3(x):\n")
        self.assertIn("return x", self.record["answer"])

    def test_wrong_name_inline_body_missing_lf_and_empty_splits_rejected(self):
        for answer in ("def wrong(x):\n    pass\n", "def fn_a1b2c3(x): pass\n", "def fn_a1b2c3(x):", ""):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                header_records([{**self.record, "answer": answer}])
        with self.assertRaises(ValueError):
            header_records([])

    def test_header_name_and_exact_response_are_separate_from_teacher_forcing(self):
        record = header_records([self.record])[0]
        expected = list(record["function"].encode())
        item = score_header(record, "def fn_a1b2c3(y):\n", expected)
        self.assertTrue(item["autonomous_header_name_correct"])
        self.assertFalse(item["exact_header_response"])
        self.assertTrue(item["teacher_forced_complete_name"])
        wrong = score_header(record, "def wrong(x):\n", expected)
        self.assertFalse(wrong["autonomous_header_name_correct"])
        self.assertTrue(wrong["teacher_forced_complete_name"])
        with self.assertRaises(ValueError):
            score_header(record, "", [])


if __name__ == "__main__":
    unittest.main()
