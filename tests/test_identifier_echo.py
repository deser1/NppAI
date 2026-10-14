import unittest

from scripts.train_identifier_echo import echo_records, encode_echo, score_echo


class IdentifierEchoTests(unittest.TestCase):
    def setUp(self):
        self.source = [{"id": "first", "function": "fn_a1b2c3"},
                       {"id": "style2", "function": "fn_a1b2c3"},
                       {"id": "next", "function": "fn_z9y8x7"}]

    def test_one_example_per_training_name_without_source_mutation(self):
        records = echo_records(self.source)
        self.assertEqual([r["id"] for r in records], ["first", "next"])
        self.assertEqual(len(self.source), 3)
        self.assertEqual(records[0]["prompt"], "Copy identifier: fn_a1b2c3")
        self.assertEqual(records[0]["answer"], "fn_a1b2c3\n")

    def test_response_alignment_masks_prompt_and_supervises_eos(self):
        record = echo_records(self.source)[0]
        x, y, start = encode_echo(record)
        self.assertEqual(bytes(y[start:-1]).decode("ascii"), record["answer"])
        self.assertEqual(y[:start], [-100] * start)
        self.assertEqual(y[-1], 0)
        self.assertEqual(x[start], ord("\n"))
        self.assertEqual(len(x), len(y))

    def test_exact_response_does_not_count_scaffolding_or_missing_newline(self):
        record = echo_records(self.source)[0]
        predictions = list(record["function"].encode("ascii"))
        correct = score_echo(record, record["answer"], predictions)
        self.assertTrue(correct["exact_response"])
        for output in (record["function"], "def " + record["answer"], "other\n"):
            item = score_echo(record, output, predictions)
            self.assertFalse(item["exact_response"])
            self.assertTrue(item["teacher_forced_complete_name"])

    def test_invalid_names_empty_splits_and_overflow_rejected(self):
        for records in ([], [{"id": "bad", "function": "other"}]):
            with self.assertRaises(ValueError):
                echo_records(records)
        record = echo_records(self.source)[0]
        with self.assertRaises(ValueError):
            encode_echo(record, 10)
        with self.assertRaises(ValueError):
            encode_echo({**record, "prompt": "bad\0prompt"})
        with self.assertRaises(ValueError):
            score_echo(record, "", [])


if __name__ == "__main__":
    unittest.main()
