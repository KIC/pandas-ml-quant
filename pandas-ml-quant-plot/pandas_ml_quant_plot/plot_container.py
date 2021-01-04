import pandas as pd
import numpy as np
from matplotlib.axis import Axis

from pandas_ml_quant_plot.plot_utils import color_positive_negative


class PlotContainer(object):

    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.plots = []

    def candlestick(self, open='Open', high='High', low='Low', close='Close', pos_color='green', neg_color='red', **kwargs):
        def plot(df, ax, index_slice):
            colors = color_positive_negative(df, open, close, pos_color, neg_color).iloc[index_slice]

            fhl = df._[[low, high]].iloc[index_slice]
            ax.vlines(fhl.index, fhl.iloc[:, 0], fhl.iloc[:, 1], colors=colors, **kwargs)

            foc  = df._[[open, close]].iloc[index_slice]
            ax.vlines(foc.index, foc.iloc[:, 0], foc.iloc[:, 1], lw=4, colors=colors,**kwargs)

        self.plots.append(plot)

    def line(self, *columns, **kwargs):
        def plot(df, ax, index_slice):
            ax.plot(df._[list(columns)].iloc[index_slice], **kwargs)

        self.plots.append(plot)

    def bar(self, columns, **kwargs):
        def plot(df, ax, index_slice):
            f = df._[list(columns)].iloc[index_slice]
            for col in f.columns:
                # TODO calculate line width, eventaully colors
                ax.vlines(f.index, np.zeros(len(f)), f[col], **kwargs)

        self.plots.append(plot)

    def plot(self, *columns, **kwargs):
        pass

    def render(self, ax: Axis, index_slice):
        ax.axes.xaxis
        if isinstance(ax, np.ndarray):
            # fixme, implement histogram plot
            pass

        for p in self.plots:
            p(self.parent.df, ax, index_slice)

    def __gt__(self, other):
        # print(">", other)
        self.parent.plot_dist = True
        return other