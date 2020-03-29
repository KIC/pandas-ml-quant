from unittest import TestCase

from pandas_ml_quant_test.config import DF_TEST, DF_TEST_MULTI
import pandas_ml_quant
import numpy as np

print(pandas_ml_quant.__version__)


class TestBackTest(TestCase):

    def test_crossing_sma_strategy(self):
        df = DF_TEST["Close"].copy()

        bt = df.q.ta_sma(20)\
               .q.ta_cross(df.q.ta_sma(60))\
               .q.ta_backtest(df, lambda sig: (sig, 1 if sig > 0 else -1))

        print(bt)
        np.testing.assert_almost_equal(151.908073, bt["net"].iloc[-1])

    def test_markowitz(self):
        df = DF_TEST_MULTI.copy()
        # portfolios = df.ml["Close"]\
        #     .q.ta_markowitz(return_period=20)\
        #     .q.backtest(df.ml["Close"], lambda weights: weights)  # FIXME allow to backtest multiple assets
        #
        # print(portfolios)
