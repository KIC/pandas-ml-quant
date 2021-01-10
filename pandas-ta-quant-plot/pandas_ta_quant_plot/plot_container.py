from copy import deepcopy

import pandas as pd
import numpy as np
from matplotlib.axis import Axis

from pandas_ta_quant_plot.plots import Hist
from pandas_ta_quant_plot.plot_utils import color_positive_negative


class PlotContainer(object):

    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.potential_hist_source = None
        self.plots = []

    def candlestick(self, open='Open', high='High', low='Low', close='Close', pos_color='green', neg_color='red', **kwargs):
        def plot(df, ax, index_slice):
            ax = ax[0] if isinstance(ax, np.ndarray) else ax
            colors = color_positive_negative(df, open, close, pos_color, neg_color).iloc[index_slice]

            fhl = df._[[low, high]].iloc[index_slice]
            ax.vlines(fhl.index, fhl.iloc[:, 0], fhl.iloc[:, 1], colors=colors, **kwargs)

            foc  = df._[[open, close]].iloc[index_slice]
            ax.vlines(foc.index, foc.iloc[:, 0], foc.iloc[:, 1], lw=4, colors=colors,**kwargs)

        self.plots.append(plot)
        self.potential_hist_source = close
        return self

    def line(self, *columns, **kwargs):
        def plot(df, ax, index_slice):
            ax = ax[0] if isinstance(ax, np.ndarray) else ax
            ax.plot(df._[list(columns)].iloc[index_slice], **kwargs)

        self.plots.append(plot)
        self.potential_hist_source = columns[0]
        return self

    def bar(self, column, **kwargs):
        def plot(df, ax, index_slice):
            ax = ax[0] if isinstance(ax, np.ndarray) else ax
            f = df._[list(column)].iloc[index_slice]
            for col in f.columns:
                ax.vlines(f.index, np.zeros(len(f)), f[col], **_slice_kwargs(kwargs, index_slice))

        self.plots.append(plot)
        self.potential_hist_source = column
        return self

    def bars(self, *columns):
        pass

    def plot(self, *columns, **kwargs):
        pass

    def render(self, ax: Axis, index_slice):
        for p in self.plots:
            p(self.parent.df, ax, index_slice)

    def __gt__(self, other: Hist):
        # print(">", other)
        self.parent.plot_dist = True
        col = deepcopy(self.potential_hist_source)

        def plot(df, ax, index_slice):
            ax = ax[1]
            f = df._[col].iloc[index_slice]
            ax.hist(f, orientation='horizontal', bins=other.buckets)

        self.plots.append(plot)
        self.potential_hist_source = None
        return None


def _slice_kwargs(kwargs, index_slice):
    return {k: v.iloc[index_slice] if isinstance(v, (pd.DataFrame, pd.Series)) else v for k, v in kwargs.items()}
