from typing import Any, NamedTuple, Callable

import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.utils import hexplode
from pandas_ml_utils import Summary, Model
from pandas_ml_utils.constants import TARGET_COLUMN_NAME, PREDICTION_COLUMN_NAME


class _Portfolio(NamedTuple):
    balance: float
    rebalance: bool
    weights_distance: float
    weights: np.ndarray
    trades: np.ndarray
    positions: np.ndarray


class PortfolioWeightsSummary(Summary):

    def __init__(self,
                 df: Typing.PatchedDataFrame,
                 model: Model,
                 rebalancing_lag: int = 1,
                 rebalance_after_distance: float = 0,
                 rebalance_after_draw_down: float = None,
                 rebalance_fee: Callable[[float], float] = lambda _: 0,
                 price_column: Any = 'Close',
                 **kwargs
                 ):
        super().__init__(
            df.sort_index(),
            model,
            self._plot_portfolio_and_weights,
            self._show_risk_metrics,
            layout=[[0, 0, 1]],
            **kwargs
        )
        assert TARGET_COLUMN_NAME in df, "Target Prices need to be provided via FeaturesAndLabels.target"
        self.rebalancing_lag = rebalancing_lag
        self.rebalance_after_distance = rebalance_after_distance
        self.rebalance_after_draw_down = rebalance_after_draw_down
        self.rebalance_fee = rebalance_fee
        self.price_column = price_column
        self.portfolio = self.construct_portfolio()
        self.weight_distances = np.linalg.norm(
            df[PREDICTION_COLUMN_NAME].values - np.roll(df[PREDICTION_COLUMN_NAME].values, 1),
            axis=1
        )

    def construct_portfolio(self):
        df = self.df

        def is_column(col, requested):
            # target looks like 'target', ('Open', 'spy'), ('Close', 'spy'),.. or 'target', 'Close'
            return col == requested or (isinstance(col, tuple) and requested in col)

        # prices
        trade_prices = df[TARGET_COLUMN_NAME]
        trade_prices = trade_prices[[col for col in trade_prices.columns if is_column(col, self.price_column)]].copy()
        trade_prices["$"] = 1

        # portfolio
        portfolio_weights = df[PREDICTION_COLUMN_NAME].copy()
        portfolio_weights["$"] = (1 - portfolio_weights.sum(axis=1)).clip(lower=+0.0)
        portfolio = self._construct_portfolio(portfolio_weights, trade_prices)

        # benchmark
        columns = portfolio["weights"].columns[:-1]
        bench_weights = pd.DataFrame(np.ones((len(df), len(columns))) / len(columns), index=df.index, columns=columns)
        benchmark = self._construct_portfolio(bench_weights, trade_prices.drop("$", axis=1))

        # return portfolio with benchmark
        portfolio["benchmark", "1/N"] = benchmark["agg", "balance"]
        return portfolio

    def _construct_portfolio(self, weights, prices):
        current_weights = np.zeros(weights.shape[1])
        trades = np.zeros(weights.shape[1])
        lots = np.zeros(weights.shape[1])
        max_balance = 1
        positions = []
        balance = 1

        for i, (idx, target_weights) in enumerate(weights.shift(self.rebalancing_lag).iterrows()):
            target_weights = target_weights.values
            trade_prices = prices.loc[idx].values
            rebalanced = False

            # calculate current portfolio balance
            if lots.sum() != 0: balance = lots @ trade_prices
            weights_distance = np.linalg.norm(current_weights - target_weights)

            # check re-balancing trigger
            trigger_rebalance = self.rebalance_after_distance <= 0 or weights_distance > self.rebalance_after_distance
            if self.rebalance_after_draw_down is not None and not trigger_rebalance:
                trigger_rebalance = (balance / max_balance) < (1 - self.rebalance_after_draw_down)
                if trigger_rebalance:
                    max_balance = balance

            # re-balance portfolio
            if (i > 1 and trigger_rebalance) or i == 1:
                new_lots = (balance * target_weights) / trade_prices * (1 - self.rebalance_fee(balance))
                trades = np.round(new_lots, 10) - np.round(lots, 10)
                lots = new_lots
                rebalanced = True

            # then add the current balance to a list such that we can calculate the performance over time
            max_balance = max(max_balance, balance)
            current_weights = (lots * trade_prices) / balance
            positions.append(_Portfolio(balance, rebalanced, weights_distance, current_weights, trades, lots))

        portfolio = pd.DataFrame(positions, index=weights.index)
        portfolio.columns = pd.MultiIndex.from_product([["agg"], portfolio.columns])
        portfolio = hexplode(portfolio, ("agg", "weights"), pd.MultiIndex.from_product([["weights"], weights.columns.tolist()]))
        portfolio = hexplode(portfolio, ("agg", "positions"), pd.MultiIndex.from_product([["positions"], weights.columns.tolist()]))
        portfolio = hexplode(portfolio, ("agg", "trades"), pd.MultiIndex.from_product([["trades"], weights.columns.tolist()]))
        return portfolio

    def _plot_portfolio_and_weights(self, *args, figsize=(20, 20), **kwargs):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, gridspec_kw={"height_ratios": [3, 1, 3, 1]}, sharex=True, figsize=figsize)
        p = self.portfolio[1:]

        # plot performance graphs
        p["agg", "balance"].plot(ax=ax[0], label="Portfolio")
        p["agg", "balance"].ta.draw_down()["dd"].plot(ax=ax[1], label="Portfolio")

        # plot draw down
        p["benchmark", "1/N"].plot(ax=ax[0], label="1/N", color="silver")
        p["benchmark", "1/N"].ta.draw_down()["dd"].plot(ax=ax[1], label="1/N", color="silver")
        ax[0].legend(loc="upper left")

        # plot weight distribution
        p["weights"].plot.area(ax=ax[2])
        ax[2].legend(loc="upper left")
        ax[3].bar(x=p.index[1:], height=p["agg", "weights_distance"].values[1:])

        return fig

    def _show_risk_metrics(self, *args, **kwargs):
        p = self.portfolio["agg", "balance"][2:]
        b = self.portfolio["benchmark", "1/N"][2:]

        def performance(s):
            return (s[-1] / s[0] - 1) * 100

        def cvar(s, confidence=0.95):
            return s.pct_change().sort_values()[:int((1 - confidence) * len(s))].values.mean() * 100

        def sharpe_ratio(s, rf=0):
            return (s.pct_change().mean() - rf) / s.std()

        return pd.DataFrame(
            [{"Performance": performance(p), "CVaR95": cvar(p), "Sharpe": sharpe_ratio(p)},
             {"Performance": performance(b), "CVaR95": cvar((b)), "Sharpe": sharpe_ratio(b)}],
            index=["Portfolio", "1/N"]
        ).T.style.set_precision(3)
