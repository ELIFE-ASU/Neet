import unittest


class TestCanary(unittest.TestCase):
    def test_add_one_two(self):
        self.assertEqual(3, 1 + 2)
