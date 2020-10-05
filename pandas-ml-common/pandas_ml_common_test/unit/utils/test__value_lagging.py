from unittest import TestCase

import numpy as np

from pandas_ml_common import pd
from pandas_ml_common.utils.column_lagging_utils import lag_columns

from pandas_ml_common_test.config import TEST_DF


class TestRegressionLagging(TestCase):

    def test_auto_regression(self):
        df = TEST_DF.copy()
        rnn = lag_columns(df[["Close", "Volume"]], [1, 2]).dropna()
        self.assertEqual(6823, len(rnn))

    def test_auto_regression_simple(self):
        df = TEST_DF.copy()
        rnn = lag_columns(df[["Close", "Volume"]], [1, 2])

        np.testing.assert_array_almost_equal(
            np.array([[df["Close"].iloc[-2], df["Close"].iloc[-3],
                       df["Volume"].iloc[-2], df["Volume"].iloc[-3]]]),
            rnn[-1:].values, 0.0001)

    def test_lag_smoothing_nan(self):
        """given"""
        df = pd.DataFrame({"featureA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        #                               1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # original
        #                                  1, 2, 3, 4, 5, 6, 7, 8, 9]   # lag 1
        #                                        1, 2, 3, 4, 5, 6, 7]   # lag 1 + shift 2
        #                                        ^                      # this is where the df starts

        """when lag smoothing is enabled using shift (which is introducing nan into the data frame)"""
        rnn = lag_columns(df[["featureA"]],
                          feature_lags=[0, 1],
                          lag_smoothing={1: lambda df: df["featureA"].shift(2)}).dropna()

        """then"""
        self.assertAlmostEqual(rnn[0, "featureA"].iloc[0], 4)
        self.assertAlmostEqual(rnn[1, "featureA"].iloc[0], 1.0)
        self.assertAlmostEqual(rnn[0, "featureA"].iloc[-1], 10)
        self.assertAlmostEqual(rnn[1, "featureA"].iloc[-1], 7.0)
