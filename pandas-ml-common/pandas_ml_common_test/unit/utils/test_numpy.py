from unittest import TestCase
import numpy as np

from pandas_ml_common.utils.numpy_utils import get_buckets


class TestNumpyUtils(TestCase):

    def test_bucketing(self):
        """given"""
        borders = np.arange(10)

        """when"""
        tuples = get_buckets(borders)

        """then"""
        self.assertEqual(
            str(tuples),
            str([(np.nan, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0),(8.0, 9.0), (9.0, np.nan)])
        )

