from unittest import TestCase

from pandas_ta_quant.technical_analysis import ta_gkyz_volatility, ta_cc_volatility, ta_repeat, ta_apply, ta_resample
from pandas_ta_quant_test.config import DF_TEST, DF_TEST_MULTI_ROW_MULTI_COLUMN
import numpy as np


class TestMeta(TestCase):

    def test_repeat(self):
        df = DF_TEST[-100:]

        def vol_ratio_multi_periods(df, param):
            return (ta_gkyz_volatility(df, period=param) / ta_cc_volatility(df["Close"], period=param) - 1)\
                .rename(f"{param}")

        result = ta_repeat(df, vol_ratio_multi_periods, range(2, 10), multiindex="HF/RF Vola Ratio")
        #print(result)

        self.assertEqual(result.shape, (100, 8))
        self.assertEqual(df.index.to_list(), result.index.to_list())

    def test_apply(self):
        df = DF_TEST_MULTI_ROW_MULTI_COLUMN
        res = ta_apply(df, lambda x: (np.min(x.values), np.max(x.values)))
        self.assertEqual((7670, 4), res.shape)
        self.assertEqual(df.index.to_list(), res.index.to_list())

    def test_rolling_apply(self):
        df = DF_TEST_MULTI_ROW_MULTI_COLUMN
        res = ta_apply(df, lambda x: (np.min(x.values), np.max(x.values)), period=10, columns=['High', 'Low'])
        self.assertEqual((7670, 4), res.shape)
        self.assertEqual(df.index.to_list(), res.index.to_list())

    def test_resample(self):
        df = DF_TEST_MULTI_ROW_MULTI_COLUMN
        res = ta_resample(df, list, 'W')
        self.assertEqual((1592, 14), res.shape)
