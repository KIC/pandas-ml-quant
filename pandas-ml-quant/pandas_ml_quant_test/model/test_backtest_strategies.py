from functools import partial
from unittest import TestCase

import numpy as np

import pandas_ml_quant
from pandas_ml_common.sampling.splitter import duplicate_data
from pandas_ml_quant.model.rolling.minimum_variance import MarkowitzModel
from pandas_ml_quant.model.summary.portfolio_weights_summary import PortfolioWeightsSummary
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
        df = DF_TEST_MULTI._["Close"][-100:]

        with df.model() as m:
            fit = m.fit(**MarkowitzModel().to_fitter_kwargs())

        bt = df.model.backtest(fit.model)
        self.assertAlmostEqual(1.11254, bt.portfolio["agg", "balance"].iloc[-1], 4)

        print(bt._repr_html_())

    # TODO provide abstract model where we can provide buy/sell siglals or something
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

