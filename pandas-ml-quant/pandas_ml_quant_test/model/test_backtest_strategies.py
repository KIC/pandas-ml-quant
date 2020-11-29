from functools import partial
from unittest import TestCase

import pandas_ml_quant
from pandas_ml_common.sampling.splitter import duplicate_data
from pandas_ml_quant.model.rolling.minimum_variance import MarkowitzModel
from pandas_ml_quant.model.summary.portfolio_weights_summary import PortfolioWeightsSummary
from pandas_ml_quant.technichal_analysis import ta_mean_returns
from pandas_ml_quant_test.config import DF_TEST, DF_TEST_MULTI
from pandas_ml_utils import LambdaModel, FeaturesAndLabels

print(pandas_ml_quant.__version__)


class TestBackTest(TestCase):

    def test_crossing_sma_strategy(self):
        df = DF_TEST[["Close"]].copy()

        with df.model() as m:
            fit = m.fit(
                LambdaModel(
                    lambda x: x.values,
                    FeaturesAndLabels(
                        features=[lambda df: (df["Close"].ta.sma(20) > df["Close"].ta.sma(60)).astype(int).rename("weight")],  # long (only) if 20 is over 60
                        labels=["Close"],  # we need a label which is just not relevant fo the model itself
                        targets=["Close"]  # needed for the summary
                    ),
                    summary_provider=partial(PortfolioWeightsSummary, rebalance_after_distance=0.000001),
                ),
                splitter=duplicate_data()
            )

        bt = df.model.backtest(fit.model)
        # print(bt._repr_html_())

        self.assertGreater(bt.portfolio["agg", "balance"].iloc[-1], 3)
        self.assertAlmostEqual(fit.training_summary.portfolio["agg", "balance"].iloc[-1], bt.portfolio["agg", "balance"].iloc[-1])
        self.assertEqual(fit.training_summary.portfolio["agg", "rebalance"].sum(), 126)
        self.assertEqual(fit.training_summary.portfolio["agg", "rebalance"].sum(), bt.portfolio["agg", "rebalance"].sum())

    def test_markowitz_strategy(self):
        df = DF_TEST_MULTI._["Close"]['2019-08-01':'2019-12-31']

        with df.model() as m:
            fit = m.fit(**MarkowitzModel(long_only=True).to_fitter_kwargs())

        bt = df.model.backtest(fit.model)
        self.assertAlmostEqual(1.03070, bt.portfolio["agg", "balance"].iloc[-1], 4)

        # print(bt._repr_html_())

    def test_markowitz_strategy_with_shorting(self):
        df = DF_TEST_MULTI._["Close"][-100:]

        with df.model() as m:
            fit = m.fit(**MarkowitzModel(
                returns_estimator=partial(ta_mean_returns, period=2),
                risk_aversion=0.99,
                long_only=False).to_fitter_kwargs())

        bt = df.model.backtest(fit.model)
        self.assertAlmostEqual(1.07992, bt.portfolio["agg", "balance"].iloc[-1], 4)

    def test_markowitz_strategy_full_frame(self):
        df = DF_TEST_MULTI['2019-08-01':'2019-12-31']

        with df.model() as m:
            fit = m.fit(**MarkowitzModel(long_only=True).to_fitter_kwargs())

        bt = df.model.backtest(fit.model)
        self.assertAlmostEqual(1.03070, bt.portfolio["agg", "balance"].iloc[-1], 2)

        bt2 = df.model.backtest(fit.model, lambda df, m: PortfolioWeightsSummary(df, m, rebalance_after_distance=0.1))
        self.assertAlmostEqual(1.03070, bt2.portfolio["agg", "balance"].iloc[-1], 2)