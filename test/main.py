import unittest


class TestImplementations(unittest.TestCase):

    def setUp(self):
        self.dummy = 1

    def test_dummy(self):
        self.assertEqual(self.dummy, 1)
