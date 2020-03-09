from unittest import TestCase

import numpy as np

from pandas_ml_quant.indicators.auto_regression import auto_regression
from pandas_ml_quant_test.config import DF_TEST


class TestRegressionLagging(TestCase):

    def test_auto_regression_simple(self):
        df = DF_TEST.copy()
        regressed, min_needed_data = auto_regression(df[["Close", "Volume"]], [1, 2])

        self.assertEqual(2, min_needed_data)
        np.testing.assert_array_almost_equal(
            np.array([[df["Close"].iloc[-2], df["Close"].iloc[-3],
                       df["Volume"].iloc[-2], df["Volume"].iloc[-3]]]),
            regressed[-1:].values, 0.0001)

    # FIXME test smoothing
