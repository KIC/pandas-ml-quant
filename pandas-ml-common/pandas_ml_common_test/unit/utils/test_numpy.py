from unittest import TestCase
import numpy as np

from pandas_ml_common.utils.numpy_utils import get_buckets, CircularBuffer, one_hot


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

    def test_circular_buffer(self):
        """given"""
        b = CircularBuffer(10)
        b2 = CircularBuffer(10)

        """when"""
        for i in range(20):
            b.append(i)
            if i < 5:
                b2.append(i)

        """then"""
        np.testing.assert_array_equal(b.buffer, np.arange(10, 20).astype(float))
        np.testing.assert_array_equal(b.get_filled(), np.arange(10, 20).astype(float))
        np.testing.assert_array_equal(b2.get_filled(), np.arange(0, 5).astype(float))

    def test_one_hot(self):
        """given integer numbers"""
        x = np.arange(5)

        """when one hot encoded"""
        ohx = one_hot(x, None)

        """then x is one hot encoded"""
        np.testing.assert_array_equal(ohx, np.eye(5))