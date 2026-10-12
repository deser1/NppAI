import unittest

import torch
import torch.nn.functional as F

from scripts.train_broader_instruction_experiment import identifier_weights, weighted_response_loss


class IdentifierTrainingTests(unittest.TestCase):
    def setUp(self):
        self.record = {"function": "fn_a9z", "prompt": "Implement fn_a9z(x) returning x.",
                       "answer": "def fn_a9z(x):\n    return x\n"}

    def test_weighted_targets_decode_to_only_the_response_identifier(self):
        _, y, weights = identifier_weights(self.record, 256, 4)
        selected = y[weights == 4].tolist()
        self.assertEqual(bytes(selected).decode(), "fn_a9z")
        self.assertTrue(torch.all(weights[y == -100] == 0))
        self.assertEqual(int(y[-1]), 0)  # EOS remains a supervised target.
        self.assertEqual(float(weights[-1]), 1)

    def test_unit_weight_matches_original_loss_and_ignores_padding(self):
        _, y, weights = identifier_weights(self.record, 256, 1)
        torch.manual_seed(3)
        logits = torch.randn(1, len(y), 256)
        original = F.cross_entropy(logits.reshape(-1, 256), y, ignore_index=-100)
        torch.testing.assert_close(weighted_response_loss(logits, y, weights), original)
        padded_y = torch.cat((y, torch.tensor([-100, -100])))
        padded_weights = torch.cat((weights, torch.zeros(2)))
        padded_logits = torch.cat((logits, torch.full((1, 2, 256), 10000.0)), dim=1)
        torch.testing.assert_close(weighted_response_loss(padded_logits, padded_y, padded_weights), original)

    def test_wrong_response_header_is_rejected(self):
        self.record["answer"] = "def other(x):\n    return x\n"
        with self.assertRaises(ValueError):
            identifier_weights(self.record, 256, 4)


if __name__ == "__main__":
    unittest.main()
