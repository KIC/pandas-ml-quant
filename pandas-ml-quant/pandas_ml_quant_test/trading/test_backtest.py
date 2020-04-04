from unittest import TestCase

from pandas_ml_quant_test.config import DF_TEST, DF_TEST_MULTI
import pandas_ml_quant
import numpy as np

print(pandas_ml_quant.__version__)


class TestBackTest(TestCase):

    def test_crossing_sma_strategy(self):
        df = DF_TEST["Close"].copy()

        bt = df.ta.sma(20)\
               .ta.cross(df.ta.sma(60))\
               .ta.backtest(df, lambda sig: (sig, 1 if sig > 0 else -1))

        print(bt)
        np.testing.assert_almost_equal(151.908073, bt["net"].iloc[-1])

    def test_crossing_pairs_strategy(self):
        df = DF_TEST_MULTI._["Close"].copy()
        correlation = df["Close", "spy"].rolling(60).corr(df["Close", "gld"])
        signal = correlation\
            .to_frame()\
            .apply(lambda v: [-1, 1] if v[0] < -0.70 else ([1, -1] if v[0] > 0.70 else [0, 0]),
                   result_type='expand',
                   axis=1)

        porfolios = signal.ta.backtest(df, lambda sig: (sig, 10 * sig))
        print(porfolios)

    def test_markowitz(self):
        df = DF_TEST_MULTI._["Close"].copy()
        weights = df.ta.markowitz(return_period=20)

        prices = df.loc[weights.index]
        weights.columns = prices.columns
        shares_for_one_dollar = weights / prices

        portfolios = shares_for_one_dollar.ta.backtest(df, lambda shares: shares * 100, lambda value: -0.01)

        print(portfolios)
        np.testing.assert_almost_equal(225.853351, portfolios["Close,spy", "net"].iloc[-1])
        np.testing.assert_almost_equal(284.032017, portfolios["Close,gld", "net"].iloc[-1])
