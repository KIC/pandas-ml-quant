from unittest import TestCase

import pandas as pd

from pandas_ml_quant.model.rolling import MinVarianceModel
from pandas_ml_quant_test.config import DF_TEST, DF_TEST_MULTI
from pandas_ml_utils import RegressionSummary
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME


class TestRollingModel(TestCase):

    def test_rolling_model(self):
        df = DF_TEST[-50-30-1:-1]

        with df.model() as m:
            from pandas_ml_quant import FeaturesAndLabels, SkModel
            from pandas_ml_quant.model import RollingModel
            from sklearn.neural_network import MLPRegressor

            skmodel = SkModel(
                MLPRegressor((25, 10), max_iter=500, shuffle=False),
                FeaturesAndLabels(
                    features=[lambda df: df["Close"].pct_change().ta.rnn(30)],
                    labels=[lambda df: df["Close"].pct_change().shift(-1)],
                ),
                summary_provider=RegressionSummary
            )

            rm = RollingModel(skmodel, 30, 5)
            fit = m.fit(**rm.to_fitter_kwargs())

        self.assertEqual(4, rm.cross_validation._nr_of_splits)
        self.assertEqual(20 - 1, len(fit.test_summary.df))  # minus one for the forecast!
        self.assertFalse(fit.test_summary.df[PREDICTION_COLUMN_NAME].isnull().values.any())

    def test_rolling_model_portfolio_construction(self):
        df = DF_TEST_MULTI[-100:]

        with df.model() as m:
            fit = m.fit(**MinVarianceModel().to_fitter_kwargs())

        print(df.index[-1], fit.training_summary.df.index[-1])
        print(fit._repr_html_())

        self.assertEqual(df.index[-1], fit.training_summary.df.index[-1])
        self.assertEqual(df.index[-1], fit.test_summary.df.index[-1])

        print()

    def test_rolling_model_multi_index_row(self):
        df = pd.concat([DF_TEST[-50-30-1:-1].add_multi_index("SPY", axis=0), DF_TEST[-52-30-1:-1].add_multi_index("GLD", axis=0)], axis=0)

        with df.model() as m:
            from pandas_ml_quant import FeaturesAndLabels, SkModel
            from pandas_ml_quant.model import RollingModel
            from sklearn.neural_network import MLPRegressor

            skmodel = SkModel(
                MLPRegressor((25, 10), max_iter=500, shuffle=False),
                FeaturesAndLabels(
                    features=[lambda df: df["Close"].pct_change().ta.rnn(30)],
                    labels=[lambda df: df["Close"].pct_change().shift(-1)],
                ),
                summary_provider=RegressionSummary
            )

            rm = RollingModel(skmodel, 30, 5)
            fit = m.fit(**rm.to_fitter_kwargs(partition_row_multi_index=True))

        self.assertEqual(rm.cross_validation._nr_of_splits, 10)
        self.assertIsInstance(fit.test_summary.df.index, pd.MultiIndex)
        self.assertIsInstance(fit.training_summary.df.index, pd.MultiIndex)

    def test_rolling_model_retraining(self):
        #  TODO solve retrain problem
        pass

# TODO implement
#  * fix and finalize this test
#  now all todos left are new features -> not part of 0.2.0

