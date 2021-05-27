from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ta_quant._utils import index_of_bucket, rolling_apply


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

    def test_roll_apply(self):
        s = pd.Series([1,2,3,4,5])

        m1 = rolling_apply(s, 3, lambda x: x.mean(), "foo")
        np.testing.assert_array_almost_equal(m1.values.reshape(len(s)), [np.nan, np.nan, 2, 3, 4])

        m4 = rolling_apply(s, 4, lambda x: x.mean(), "foo")
        np.testing.assert_array_almost_equal(m4.values.reshape(len(s)), [np.nan, np.nan, np.nan, 2.5, 3.5])

        m5 = rolling_apply(s, 5, lambda x: x.mean(), "foo")
        np.testing.assert_array_almost_equal(m5.values.reshape(len(s)), [np.nan, np.nan, np.nan, np.nan, 3])

        m2 = rolling_apply(s, 3, lambda x: x.mean(), "bar", center=True)
        np.testing.assert_array_almost_equal(m2.values.reshape(len(s)), [np.nan, 2, 3, 4, np.nan])

        m3 = rolling_apply(s, 5, lambda x: x.mean(), "baz", center=True)
        np.testing.assert_array_almost_equal(m3.values.reshape(len(s)), [np.nan, np.nan, 3, np.nan, np.nan])