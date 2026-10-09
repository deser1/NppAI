import unittest
from solution import sum_positive


class TestSumPositive(unittest.TestCase):
    def test_mixed(self):
        self.assertEqual(sum_positive([-2, 3, 0, 4]), 7)

    def test_empty(self):
        self.assertEqual(sum_positive([]), 0)

    def test_negative(self):
        self.assertEqual(sum_positive([-9, -1]), 0)


if __name__ == "__main__":
    unittest.main()
