from unittest import TestCase

import numpy as np

from pandas_ml_quant import pd
from pandas_ml_quant.technical_analysis_patch import ta_rnn, ta_normalize_row
from pandas_ml_quant_test.config import DF_TEST
from pandas_ta_quant.technical_analysis import ta_rescale


class TestRegressionLagging(TestCase):

    def test_auto_regression(self):
        df = DF_TEST.copy()
        rnn = ta_rnn(df[["Close", "Volume"]], [1, 2])

        self.assertEqual(6761, len(rnn))


    def test_auto_regression_simple(self):
        df = DF_TEST.copy()
        rnn, min_needed_data = ta_rnn(df[["Close", "Volume"]], [1, 2],
                                      return_min_required_samples=True)

        self.assertEqual(2, min_needed_data)
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
        rnn, min_needed_data = ta_rnn(df[["featureA"]],
                                      feature_lags=[0, 1],
                                      lag_smoothing={1: lambda df: df["featureA"].shift(2)},
                                      return_min_required_samples=True)


        len_features = 10 - 1 - 2

        """then"""
        self.assertEqual(len(rnn), len_features)
        self.assertAlmostEqual(rnn[0, "featureA"].iloc[0], 4)
        self.assertAlmostEqual(rnn[1, "featureA"].iloc[0], 1.0)
        self.assertAlmostEqual(rnn[0, "featureA"].iloc[-1], 10)
        self.assertAlmostEqual(rnn[1, "featureA"].iloc[-1], 7.0)

    # test normaluisation of autiregressors
    def test_rescale_embedded(self):
        df = DF_TEST[["Close", "High"]][-5:].ta.rnn(3).copy()
        rows = ta_rescale(df, (0, 1), axis=1)
        print(df)
        print(rows)

        self.assertEqual(3, np.argmax(rows.iloc[-1]))

    def test_normalize_row_multi_column(self):
        df = ta_rnn(pd.DataFrame({"a": np.arange(10), "b": np.arange(10)}), 5)

        mm01 = ta_normalize_row(df, 'minmax01', level=1)
        mm11 = ta_normalize_row(df, 'minmax-11', level=1)
        std = ta_normalize_row(df, 'standard', level=1)
        uni = ta_normalize_row(df, 'uniform', level=1)

        np.testing.assert_array_equal(mm01.xs("a", level=1, axis=1), mm01.xs("b", level=1, axis=1))
        np.testing.assert_array_equal(mm11.xs("a", level=1, axis=1), mm11.xs("b", level=1, axis=1))
        np.testing.assert_array_almost_equal(std.xs("a", level=1, axis=1), std.xs("b", level=1, axis=1))
        np.testing.assert_array_equal(uni.xs("a", level=1, axis=1), uni.xs("b", level=1, axis=1))
