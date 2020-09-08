from pandas_ml_common import Typing, inner_join
import pandas as pd
import numpy as np


class PortfolioFromWeights(object):

    def __init__(self, weights: Typing.PatchedDataFrame, prices: Typing.PatchedDataFrame, rebalance_threshold=0.01):
        super().__init__()
        self.rebalance_threshold = rebalance_threshold
        self.prices = prices
        self.raw_weights = weights
        self.delta_weights = weights - weights.shift(1).fillna(0)
        self.rebalancing = self._rebalancing(rebalance_threshold)
        self.weights = self._smooth_weights() if rebalance_threshold is not None else weights
        self.fractions = self._calculate_fractions()
        self.portfolio_return = self._evaluate_return()
        self.performance = (self.portfolio_return + 1).cumprod()

    def _rebalancing(self, rebalance_threshold):
        max_diff = self.delta_weights.apply(lambda r: np.diff(r).max(), axis=1)
        return pd.Series(max_diff >= abs(rebalance_threshold), name='Rebalancing')

    def _smooth_weights(self):
        weights = self.raw_weights.copy()
        weights[np.invert(self.rebalancing)] = None
        return weights.fillna(method='ffill')

    def _calculate_fractions(self):
        # the weights we get at time t can only be executed at t+1
        # if you have 100$ you can buy 100 / price fractions of a share
        weights = self.weights.shift(1).fillna(0)
        fractions = weights / self.prices
        return fractions

    def _evaluate_return(self):
        # the weights we get at time t can only be executed at t+1
        #   -> weights = self.weights.shift(1).fillna(0)
        # and for all weight increments the return at time t is 0
        # only for all unchanged weights or for all weight decrements the return is close / prev_close - 1
        # this means we need to use t-2 weights to calculate the portfolio returns of time t
        weights = self.weights.shift(2).fillna(0)
        returns = self.prices.pct_change().fillna(0)
        weight_returns = inner_join(weights, returns, prefix_left='weights', prefix='returns', force_multi_index=True)
        return weight_returns.apply(lambda x: x["weights"] @ x["returns"], axis=1)

    def latest_signal(self, tail=1, amount=1):
        fractions = (self.fractions[-tail:] * amount).join(self.rebalancing[-tail:])
        return fractions

    def target_quantity(self, amount=1, current_prices=None):
        if current_prices is None:
            current_prices = self.prices.iloc[-1].values

        return (self.weights[-1:] / np.array([current_prices])) * amount

    def plot(self, figsize=(20, 10)):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=figsize, sharex=True)
        self.performance.plot(ax=ax[0], label='Portfolio')
        self.weights.plot.area(ax=ax[1])
        ax[0].legend(loc='upper right')
        return fig, ax

    def _repr_html_(self):
        from pandas_ml_common.utils import plot_to_html_img
        return f'<img src="{plot_to_html_img(self.plot)}", width="100%" />'


