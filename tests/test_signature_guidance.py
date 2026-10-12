import unittest
from scripts.evaluate_signature_guidance import signature_prefix,score


class GuidanceTests(unittest.TestCase):
    def test_task_signature_is_preserved(self):
        self.assertEqual(signature_prefix("Implement fn_add_13(a, b) returning their sum."),"def fn_add_13(a, b):\n    ")

    def test_reject_ambiguous_or_unsafe_signatures(self):
        for task in ("Do something", "f(x) or g(y)","f(x, x)","f(x=1)","class(x)","f(for)","f(g(x))"):
            with self.assertRaises(ValueError): signature_prefix(task)

    def test_wrong_name_cannot_pass_reference_score(self):
        result=score("def old(x):\n    return x\n","def requested(x):\n    return x\n","requested")
        self.assertTrue(result["syntax"])
        self.assertFalse(result["requested_function"])
        self.assertFalse(result["reference_ast"])


if __name__=="__main__": unittest.main()
