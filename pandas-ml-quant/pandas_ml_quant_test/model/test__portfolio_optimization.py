from unittest import TestCase

import pandas as pd
import numpy as np

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
                max_short_weight=0,  # long only
                max_long_weight=1,   # one single long position is allowed
                forecast_provider=None
            )
        )

        print(weights[FEATURE_COLUMN_NAME].ML[("Close")])
        print(weights[PREDICTION_COLUMN_NAME])