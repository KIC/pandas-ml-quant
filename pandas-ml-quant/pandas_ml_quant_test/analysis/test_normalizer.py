from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_quant.technichal_analysis import ta_ma_ratio, ta_normalize_row, ta_delta_hedged_price, ta_rnn
from pandas_ml_quant_test.config import DF_TEST, DF_TEST2


class TestNormalizer(TestCase):

    def test_normalize_row(self):
        df = pd.DataFrame({"a": [1,2,3], "b": [3,6,9]})
        mm01 = ta_normalize_row(df, 'minmax01')
        mm11 = ta_normalize_row(df, 'minmax-11')
        std = ta_normalize_row(df, 'standard')
        uni = ta_normalize_row(df, 'uniform')

        np.testing.assert_array_almost_equal(mm01.values, np.stack([np.zeros(3), np.ones(3)], axis=1))
        np.testing.assert_array_almost_equal(mm11.values, np.stack([np.zeros(3) - 1, np.ones(3)], axis=1))
        np.testing.assert_array_almost_equal(std.values, np.array([[-1,  1], [0, 4], [1,  7]]))
        # since we only have 2 columns and we want the values to be unified we assume both be the same
        np.testing.assert_array_almost_equal(uni.values, np.ones((3, 2)))

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

    def test_ma_ratio(self):
        df = DF_TEST

        ma_ration_scaled = ta_ma_ratio(df)
        print(ma_ration_scaled.columns)
        self.assertTrue(True)

    def test_delta_hedged_prices(self):
        df = DF_TEST
        bench = DF_TEST2["Close"]

        self.assertEqual((3788, 6), ta_delta_hedged_price(df, bench).shape)
        self.assertEqual((3788,), ta_delta_hedged_price(df["Close"], bench).shape)
        self.assertEqual((6763, 5), ta_delta_hedged_price(df, "Open").shape)
