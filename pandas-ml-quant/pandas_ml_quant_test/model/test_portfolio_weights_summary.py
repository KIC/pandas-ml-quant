from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_quant.model.summary.portfolio_weights_summary import PortfolioWeightsSummary
from pandas_ml_quant_test.config import DF_TEST_MULTI
from pandas_ml_utils.constants import *

pd.set_option('display.max_columns', None)


class TestSummary(TestCase):
    # df = DF_TEST_MULTI[-50:]._["Close"].copy()
    # df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
    # df[PREDICTION_COLUMN_NAME, "spy"] = np.random.random(len(df))
    # df[PREDICTION_COLUMN_NAME, "gld"] = 1 - df[PREDICTION_COLUMN_NAME, "spy"]

    def test_portfolio_weights_summary_simple(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = 1
        df[PREDICTION_COLUMN_NAME, "gld"] = 0

        portfolio = PortfolioWeightsSummary(df, None, rebalance_after_distance=None).construct_portfolio()
        #print(portfolio)

        self.assertAlmostEqual(0.0030852771, portfolio["trades"].values.sum().item())
        self.assertAlmostEqual(0, portfolio["trades", "gld"].values.sum().item())
        self.assertAlmostEqual(0, portfolio["positions", "gld"].values.sum().item())
        self.assertAlmostEqual(0, portfolio["weights", "gld"].values.sum().item())

        np.testing.assert_array_almost_equal(
            portfolio[1:]["agg", "balance"].pct_change().values,
            df[TARGET_COLUMN_NAME, ("Close", "spy")][1:].pct_change().values,
            6
        )

    def test_portfolio_weights_summary_equal_weights(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = 0.5
        df[PREDICTION_COLUMN_NAME, "gld"] = 0.5

        portfolio = PortfolioWeightsSummary(df, None).construct_portfolio()
        # print(portfolio)

        np.testing.assert_array_almost_equal(
            portfolio[1:]["agg", "balance"].pct_change().values,
            np.array(
                [np.nan, 0.001116185694516, 0.006388651124503, 0.004376611760257, -0.001341899428812,
                 0.005018299107146, -0.000854496810412, 0.002815041534534, 0.002476227461891]
            ),
            4
        )

    def test_portfolio_weights_summary_equal_weights_with_distance(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = 0.5
        df[PREDICTION_COLUMN_NAME, "gld"] = 0.5

        summary = PortfolioWeightsSummary(df, None, rebalance_after_distance=0.0035)
        portfolio = summary.construct_portfolio()
        #print(portfolio["agg", "balance"].pct_change())

        self.assertEqual(4, portfolio["agg", "rebalance"].sum())
        self.assertLess(portfolio["weights"][1:].values.min(), 0.493)
        self.assertGreater(portfolio["weights"][1:].values.max(), 0.505)
        np.testing.assert_equal(
            portfolio[portfolio["agg", "rebalance"] == True]["weights"].values[:, :-1],
            np.ones((4, 2)) / 2,
            8
        )
        np.testing.assert_array_almost_equal(
            portfolio["agg", "balance"].pct_change()[portfolio["agg", "rebalance"].shift(1) == True].values,
            np.array([0.001116, 0.004377, -0.000854]),
            5
        )
        self.assertLess(
            np.abs(portfolio["agg", "balance"].pct_change()[2:].values - np.array(
                [0.001116185694516, 0.006388651124503, 0.004376611760257, -0.001341899428812,
                 0.005018299107146, -0.000854496810412, 0.002815041534534, 0.002476227461891]
            )).max(), 0.0001)

    def test_portfolio_weights_summary_equal_weights_with_distance_and_fee(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = 0.5
        df[PREDICTION_COLUMN_NAME, "gld"] = 0.5

        # one percent fee
        summary = PortfolioWeightsSummary(df, None, rebalance_after_distance=0.0035, rebalance_fee=lambda b: b * 0.002)
        portfolio = summary.construct_portfolio()
        # print(portfolio)

        np.testing.assert_array_almost_equal(
            portfolio["agg", "balance"].pct_change().values[2:],
            np.array([-0.00088605, 0.00646145, 0.00235667, -0.00133788, 0.00500606, -0.00287603, 0.00282444, 0.00245626]),
            7
        )

    def test_portfolio_weights_summary_changing_weights_lag(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = np.linspace(0.1, 1, 10)
        df[PREDICTION_COLUMN_NAME, "gld"] = 1 - np.linspace(0.1, 1, 10)

        summary = PortfolioWeightsSummary(df, None)
        portfolio = summary.construct_portfolio()
        #print(portfolio["agg", "balance"].pct_change())

        np.testing.assert_array_almost_equal(
            portfolio["agg", "balance"].pct_change().values[2:],
            np.array(
                [-0.010183880682612, 0.003293014229764, 0.004781345270464, -0.000544347222833, 0.005018299107146, -0.000336997711267, 0.004266129257653, 0.00054594116554]),
            7
        )

    def test_portfolio_weights_summary_changing_weights_lag_with_distance(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = np.linspace(0.1, 1, 10)
        df[PREDICTION_COLUMN_NAME, "gld"] = 1 - np.linspace(0.1, 1, 10)

        summary = PortfolioWeightsSummary(df, None, rebalance_after_distance=0.2)
        portfolio = summary.construct_portfolio()
        # print(portfolio)

        np.testing.assert_array_almost_equal(
            portfolio["agg", "balance"].pct_change().values[2:],
            np.array(
                [-0.01018388, 0.00228764, 0.00478135, 0.00025658, 0.0050183, -0.0008482, 0.00426613, 0.00117961]),
            7
        )

    def test_cash_fallback(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = 0.5
        df[PREDICTION_COLUMN_NAME, "gld"] = 0.0

        summary = PortfolioWeightsSummary(df, None, rebalance_after_distance=None)
        portfolio = summary.construct_portfolio()
        # print(portfolio)

        np.testing.assert_equal(np.repeat([0.0], 9), portfolio["weights", "gld"][1:])
        np.testing.assert_equal(np.repeat([0.5], 9), portfolio["weights", "spy"][1:])
        np.testing.assert_equal(np.repeat([0.5], 9), portfolio["weights", "$"][1:])

    def test_rebalance_draw_down(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = 1
        df[PREDICTION_COLUMN_NAME, "gld"] = 0

        portfolio = PortfolioWeightsSummary(df, None, rebalance_after_distance=100, rebalance_after_draw_down=0.005).construct_portfolio()
        print(portfolio)

        self.assertTrue(portfolio.loc["2020-02-07"]["agg", "rebalance"].item())
        self.assertEqual(1, portfolio["agg", "rebalance"][2:].values.sum())

    def test_short(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = -1
        df[PREDICTION_COLUMN_NAME, "gld"] = 0

        portfolio = PortfolioWeightsSummary(df, None, rebalance_after_distance=None).construct_portfolio()

        self.assertAlmostEqual(0, portfolio["trades", "gld"].values.sum().item())
        self.assertAlmostEqual(0, portfolio["positions", "gld"].values.sum().item())
        self.assertAlmostEqual(0, portfolio["weights", "gld"].values.sum().item())

        np.testing.assert_array_almost_equal(
            portfolio[1:]["agg", "balance"].pct_change().values,
            df[TARGET_COLUMN_NAME, ("Close", "spy")][1:].pct_change().values * -1,
            6
        )

    def test_portfolio_weights_summary_equal_weights_short(self):
        df = DF_TEST_MULTI[-10:]._["Close"].copy()
        df.columns = pd.MultiIndex.from_product([[TARGET_COLUMN_NAME], df.columns.tolist()])
        df[PREDICTION_COLUMN_NAME, "spy"] = -0.5
        df[PREDICTION_COLUMN_NAME, "gld"] = -0.5

        portfolio = PortfolioWeightsSummary(df, None).construct_portfolio()
        # print(portfolio)

        np.testing.assert_array_almost_equal(
            portfolio[1:]["agg", "balance"].pct_change().values,
            np.array(
                [np.nan, 0.001116185694516, 0.006388651124503, 0.004376611760257, -0.001341899428812,
                 0.005018299107146, -0.000854496810412, 0.002815041534534, 0.002476227461891]
            ) * -1,
            4
        )