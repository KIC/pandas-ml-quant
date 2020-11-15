from unittest import TestCase

from pandas_ml_common.sampling.splitter import duplicate_data
from pandas_ml_common.utils import wrap_row_level_as_nested_array
from pandas_ml_quant.model.rolling import MinVarianceModel
from pandas_ml_quant.model.summary.portfolio_weights_summary import PortfolioWeightsSummary
from pandas_ml_quant_test.config import DF_TEST, DF_TEST_MULTI_ROW, DF_TEST_MULTI
from pandas_ml_utils import RegressionSummary, PostProcessedFeaturesAndLabels
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
            fit2 = m.fit(skmodel, splitter=duplicate_data())

        nr_of_folds = len(list(rm.cross_validation.split(df.index[30:])))
        self.assertEqual(3, nr_of_folds)
        self.assertEqual(20 - 1, len(fit.test_summary.df))  # minus one for the forecast!
        self.assertFalse(fit.test_summary.df[PREDICTION_COLUMN_NAME].isnull().values.any())

        print(fit)
        print(fit.test_summary.df.tail())
        print(fit.test_summary.df.columns)

        with self.assertRaises(ValueError) as cm:
            # 30 for he rnn 5 we can predict but one more causes an error
            df[-30-5-1:].model.predict(fit.model)

        self.assertEqual("Model need to be re-trained!", str(cm.exception))
        print(fit2._repr_html_())
        # FIXME what if we do not have enough history in the data frame ???
        # df[-32:].model.predict(fit.model) -> fails the re-train check :-(

    def test_rolling_model_portfolio_construction(self):
        df = DF_TEST_MULTI[-100:]

        with df.model() as m:
            fit = m.fit(**MinVarianceModel(summary_provider=PortfolioWeightsSummary).to_fitter_kwargs())

        print(df.index[-1], fit.training_summary.df.index[-1])
        print(fit)
        print(fit.test_summary.df.tail())
        print(fit.test_summary.df.columns)

        print(fit._repr_html_())
        print()

    def test_rolling_model_multi_index_row(self):
        df = DF_TEST_MULTI_ROW

        with df.model() as m:
            pass

    def test_scratch(self):
        df = DF_TEST_MULTI
        fl = PostProcessedFeaturesAndLabels(
            features=["Close"],
            feature_post_processor=[lambda df: df.ta.ewma_covariance(), wrap_row_level_as_nested_array],
            labels=["Close"],
        )

        with df.model() as m:
            fwl = m.extract(fl)
            print(fwl)

    # TODO integrate the markowitz model as a rolling model
    #  pandas_ml_quant.trading.strategy.optimized.ta_markowitz
