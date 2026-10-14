import random
import unittest

from scripts.position_jitter import PositionJitter


class PositionJitterTests(unittest.TestCase):
    def setUp(self):
        self.record = {"id": "train", "function": "fn_a1b2c3", "prompt": "Implement fn_a1b2c3(x).",
                       "answer": "def fn_a1b2c3(x):\n    return x\n"}

    def test_views_preserve_names_answers_and_original_task(self):
        jitter = PositionJitter()
        for _ in range(100):
            view = jitter.apply(self.record)
            self.assertEqual(view["prompt"].strip(), self.record["prompt"])
            self.assertLessEqual(len(view["prompt"].encode()) - len(self.record["prompt"].encode()), 48)
            for key in ("id", "function", "answer"):
                self.assertEqual(view[key], self.record[key])
        self.assertEqual(self.record["prompt"], "Implement fn_a1b2c3(x).")
        self.assertEqual(jitter.metadata()["training_views"], 100)

    def test_seed_and_ordered_view_hash_are_reproducible(self):
        first, second = PositionJitter(), PositionJitter()
        self.assertEqual([first.apply(self.record) for _ in range(20)],
                         [second.apply(self.record) for _ in range(20)])
        self.assertEqual(first.metadata(), second.metadata())
        different = PositionJitter(seed=4202)
        for _ in range(20):
            different.apply(self.record)
        self.assertNotEqual(first.metadata()["ordered_view_sha256"], different.metadata()["ordered_view_sha256"])

    def test_jitter_does_not_change_the_source_batch_rng_schedule(self):
        population = [{**self.record, "id": str(i)} for i in range(40)]
        random.seed(42)
        plain = [[r["id"] for r in random.sample(population, 4)] for _ in range(10)]
        random.seed(42)
        jitter = PositionJitter()
        shifted = [[jitter.apply(r)["id"] for r in random.sample(population, 4)] for _ in range(10)]
        self.assertEqual(plain, shifted)


if __name__ == "__main__":
    unittest.main()
