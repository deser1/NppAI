import ast
import unittest
from scripts.build_instruction_dataset import build, UNSEEN


class DatasetIsolationTests(unittest.TestCase):
    def test_frozen_task_and_family_separation(self):
        data=build()
        self.assertEqual(data,build())
        self.assertEqual([len(data[k]) for k in ("train","validation","test")],[576,16,28])
        for field in ("prompt","answer","function"):
            values=[{record[field] for record in data[k]} for k in ("train","validation","test")]
            for a,b in ((0,1),(0,2),(1,2)):
                self.assertFalse(values[a]&values[b])
        self.assertFalse({record["family"] for record in data["train"]}&set(UNSEEN))
        self.assertEqual({r["family"] for r in data["test"]}&set(UNSEEN),set(UNSEEN))

    def test_reference_answers_have_requested_function(self):
        for records in build().values():
            for record in records:
                tree=ast.parse(record["answer"])
                self.assertEqual(tree.body[0].name,record["function"])

    def test_v2_uses_same_identifier_shape_in_all_splits(self):
        import re
        data=build(2)
        for records in data.values():
            for record in records:
                self.assertRegex(record["function"],r"^fn_[a-z]+_[0-9]{2}$")
        ids=[{r["function"] for r in data[k]} for k in ("train","validation","test")]
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])


if __name__=="__main__":
    unittest.main()
