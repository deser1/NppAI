import unittest

from scripts.identifier_curriculum import validate_initial_weights


class IdentifierCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.weights = {"dim": 64, "hidden": 128, "context": 256,
                        "layers": [{}, {}], "embedding": [None] * 256}

    def test_matching_checkpoint_is_accepted(self):
        validate_initial_weights(self.weights)

    def test_echo_context96_and_other_architectures_are_rejected(self):
        for field, value in (("context", 96), ("dim", 32), ("hidden", 64),
                             ("layers", [{}]), ("embedding", [None] * 512)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_initial_weights({**self.weights, field: value})


if __name__ == "__main__":
    unittest.main()
