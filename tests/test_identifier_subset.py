import unittest

from scripts.identifier_subset import select_training_records, selection_metadata


class IdentifierSubsetTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"id": str(i), "function": name} for i, name in enumerate(("a", "b", "a", "c", "b", "a"))]

    def test_keeps_all_styles_in_original_order_without_mutating_source(self):
        chosen = select_training_records(self.records, 2)
        self.assertEqual([r["id"] for r in chosen], ["0", "1", "2", "4", "5"])
        self.assertEqual(len(self.records), 6)
        self.assertEqual(selection_metadata(chosen, 2)["names"], 2)

    def test_full_selection_and_hash_are_stable(self):
        self.assertEqual(select_training_records(self.records, 3), self.records)
        self.assertEqual(select_training_records(self.records), self.records)
        self.assertEqual(selection_metadata(self.records, 3), selection_metadata(list(self.records), 3))
        self.assertNotEqual(selection_metadata(self.records, 3)["selection_sha256"],
                            selection_metadata(self.records[::-1], 3)["selection_sha256"])

    def test_invalid_limits_fail_instead_of_silently_using_all_records(self):
        for limit in (0, -1, 4, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                select_training_records(self.records, limit)


if __name__ == "__main__":
    unittest.main()
