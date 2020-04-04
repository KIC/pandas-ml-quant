import logging
from unittest import TestCase

from pandas_ml_quant.analysis.indicators import ta_zscore
from pandas_ml_quant.trading.strategy.optimized import ta_markowitz
from pandas_ml_quant_test.config import DF_TEST_MULTI
from pandas_ml_common import pd, np, inner_join

logging.basicConfig(level=logging.DEBUG)


class TestOptimizedStrategies(TestCase):

    def test__ewma_markowitz(self):
        """given"""
        df = DF_TEST_MULTI

        """when"""
        portfolios = ta_markowitz(df, return_period=20)

        """then"""
        print(portfolios)
        np.testing.assert_array_almost_equal(np.array([0.683908, 3.160920e-01]), portfolios.iloc[-4].values, 0.00001)

    def test__given_expected_returns(self):
        # _default_returns_estimator
        pass

    def test__markowitz_strategy(self):
        """given"""
        df = DF_TEST_MULTI
        df_price = df._['Close']
        df_expected_returns = df_price._["Close"].ta.macd()._["histogram"]
        df_trigger = ta_zscore(df_price['Close']).abs() > 2.0
        df_data = inner_join(df_price, df_expected_returns)
        df_data = inner_join(df_data, df_trigger, prefix='trigger')

        """when"""
        portfolios = ta_markowitz(df_data,
                                  prices='Close',
                                  expected_returns='histogram',
                                  rebalance_trigger='trigger')

        """then"""
        print(portfolios)
        np.testing.assert_array_almost_equal(np.array([1.000000, 3.904113e-07]), portfolios.iloc[-1].values, 0.00001)
