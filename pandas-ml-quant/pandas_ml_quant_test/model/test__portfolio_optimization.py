from unittest import TestCase

from pandas_ml_quant.model.portfolio_optimization import MarkowitzModel
from pandas_ml_quant_test.config import DF_TEST_MULTI
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME, FEATURE_COLUMN_NAME


class TestPortfolioOptimization(TestCase):

    def test_markowitz(self):
        df = DF_TEST_MULTI[:35].copy()

        weights = df.model.predict(
            MarkowitzModel(
                price_columns=[("spy", "Close"), ("gld", "Close")],
                expected_returns=lambda df: df.ML["Close"].ta.macd().ML["histogram_.*"].droplevel(0, axis=1),
                covariance_estimator=lambda df: df.ML["Close"].ta.ewma_covariance(reduce=True),
                add_riskless_cash_as=("CASH", "Close"),
                riskaversion=2,
                max_short_weight=1e-4,  # 0 for long only
                max_long_weight=1.0,   # one single long position is allowed
                forecast_provider=None
            )
        )

        print(weights[FEATURE_COLUMN_NAME].iloc[:, :3])
        print(weights[PREDICTION_COLUMN_NAME])

        self.assertAlmostEqual(4, weights[PREDICTION_COLUMN_NAME].sum(axis=1).sum(), 6)
        self.assertGreaterEqual(weights[PREDICTION_COLUMN_NAME].min().min(), 0)

    def test_markowitz_long_short(self):
        df = DF_TEST_MULTI[:35].copy()

        weights = df.model.predict(
            MarkowitzModel(
                price_columns=[("spy", "Close"), ("gld", "Close")],
                expected_returns=lambda df: df.ML["Close"].ta.macd().ML["histogram_.*"].droplevel(0, axis=1),
                covariance_estimator=lambda df: df.ML["Close"].ta.ewma_covariance(reduce=True),
                riskaversion=999,
                max_short_weight=-0.75,  # 0 for long only
                max_long_weight=1,   # one single long position is allowed
                forecast_provider=None
            )
        )

        print(weights[FEATURE_COLUMN_NAME].iloc[:, :2])
        print(weights[PREDICTION_COLUMN_NAME])

        self.assertAlmostEqual(4, weights[PREDICTION_COLUMN_NAME].sum(axis=1).sum(), 6)
        self.assertLessEqual(weights[PREDICTION_COLUMN_NAME].min().min(), 0)