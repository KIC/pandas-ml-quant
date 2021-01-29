from unittest import TestCase

import numpy as np

from pandas_ta_quant.technical_analysis import ta_ma_ratio, ta_returns, ta_cumret, ta_log_returns, ta_performance, \
    ta_cumlogret, ta_realative_candles
from pandas_ta_quant_test.config import DF_TEST


class TestNormalizer(TestCase):

    def test_relative_candles(self):
        df = DF_TEST[-2:].copy()
        relative = ta_realative_candles(df, volume=None)

        np.testing.assert_array_almost_equal(np.array([0.002469,  0.002021,  0.002533, -0.002829]),
                                             relative.values[-1])

    def test_ma_ratio(self):
        df = DF_TEST

        ma_ration_scaled = ta_ma_ratio(df)
        print(ma_ration_scaled.columns)
        self.assertTrue(True)

    def test_cumret(self):
        c = DF_TEST["Close"]
        perf = (1 + c.pct_change()).cumprod().rename("perf")
        cumrets = ta_cumret(ta_returns(c, 2), 2).rename("cumrets")

        results = perf.to_frame().join(cumrets).dropna()
        np.testing.assert_array_almost_equal(results["cumrets"].values[-20:-2], results["perf"].values[-20:-2])

    def test_cumlogret(self):
        c = DF_TEST["Close"]
        perf = ta_performance(c)
        logret = ta_log_returns(c)
        cumrets = ta_cumlogret(logret)

        np.testing.assert_array_almost_equal(cumrets.values, perf.values)