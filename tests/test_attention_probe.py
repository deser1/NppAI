import unittest

from scripts.attention_probe import build_model, configuration, rotate_pairs, validate_configuration

try:
    import torch
except ImportError:
    torch = None


class ProbeConfigurationTests(unittest.TestCase):
    def test_preselected_arms_and_tasks_are_strict(self):
        for arm in ("legacy1", "absolute4", "rotary4"):
            for task in ("headers", "bodies"):
                config = configuration(arm, task)
                validate_configuration(config)
                with self.assertRaises(ValueError):
                    validate_configuration({**config, "context": 128})
        with self.assertRaises(ValueError):
            configuration("unknown", "headers")


@unittest.skipIf(torch is None, "PyTorch mathematics verified in CI")
class AttentionMathTests(unittest.TestCase):
    def test_rotation_preserves_norm_and_relative_dot_products(self):
        torch.manual_seed(4)
        q, k = torch.randn(2, 4, 7, 16, dtype=torch.float64), torch.randn(2, 4, 7, 16, dtype=torch.float64)
        torch.testing.assert_close(rotate_pairs(q).norm(dim=-1), q.norm(dim=-1))
        base = rotate_pairs(q) @ rotate_pairs(k).transpose(-2, -1)
        shifted = rotate_pairs(q, 13) @ rotate_pairs(k, 13).transpose(-2, -1)
        torch.testing.assert_close(base, shifted)
        with self.assertRaises(ValueError):
            rotate_pairs(torch.randn(1, 1, 2, 3))

    def test_all_arms_are_causal_and_have_expected_logit_shape(self):
        for arm in ("legacy1", "absolute4", "rotary4"):
            torch.manual_seed(42)
            model = build_model(configuration(arm, "headers")).eval()
            with torch.no_grad():
                first = model(torch.tensor([[3, 4, 5]]))
                changed = model(torch.tensor([[3, 4, 99]]))
            self.assertEqual(tuple(first.shape), (1, 3, 256))
            torch.testing.assert_close(first[:, :2], changed[:, :2])

    def test_same_seed_initializes_shared_parameters_identically(self):
        states = []
        for arm in ("legacy1", "absolute4", "rotary4"):
            torch.manual_seed(42)
            model = build_model(configuration(arm, "headers"))
            states.append(model.state_dict())
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name])
            torch.testing.assert_close(states[0][name], states[2][name])


if __name__ == "__main__":
    unittest.main()
