from unittest import TestCase

import numpy as np

from pandas_ml_quant.utils import index_of_bucket


class TestUtils(TestCase):

    def test_index_of_bucket(self):

        a = index_of_bucket(0, np.array([1, 2, 3]))
        b = index_of_bucket(1.2, np.array([1, 2, 3]))
        c = index_of_bucket(2.1, np.array([1, 2, 3]))
        d = index_of_bucket(3, np.array([1, 2, 3]))

        self.assertEqual(0, a)
        self.assertEqual(1, b)
        self.assertEqual(2, c)
        self.assertEqual(3, d)
