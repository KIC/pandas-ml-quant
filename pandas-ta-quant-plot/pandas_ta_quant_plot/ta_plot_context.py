from collections import defaultdict
from typing import Any, Union, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector, MultiCursor

from pandas_ml_common import Typing
from pandas_ta_quant_plot.plot_container import PlotContainer
from pandas_ta_quant_plot.plot_utils import color_positive_negative


class PlotContext(object):

    def __init__(self,
                 df: Typing.PatchedDataFrame,
                 range_slider_price: str = "Close",
                 width: int = 20,
                 main_height: int = 11,
                 start: Union[Any, int] = None,
                 stop: Union[Any, int] = None,
                 h_ratio: Tuple[int] = (10, 2),
                 w_ratio: Tuple[int] = (9, 1),
                 annotate: bool = False,
                 cursor: bool = False,
                 backend='notebook'
                 ):
        self.df = df
        self.range_slider_price = range_slider_price
        self.width = width
        self.main_height = main_height
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.subset = None if start is None and stop is None else slice(start, stop)
        self.annotate = annotate
        self.cursor = cursor

        self.plots = dict()
        self.plot_dist = False
        self.widgets = defaultdict(lambda: [])
        self.fig = None
        self.ax = None

        if backend is not None:
            plt.switch_backend(backend.replace('notebook', 'nbAgg'))

    def __enter__(self):
        # plotting using a simplistic DSL like data structure
        #   with df.ta_plot() as p:
        #       p["main"].candlestick("Open", "High", "Low", "Close")
        #       p["main"].line(df["Close"].ta.sma(20))
        #       p["volume"].bar("Volume")
        #       p["macd"].plot(df["Close"].ta.macd(), "line", "line", "bar")
        #       p > "dist"
                # bring back plotting:
        return self

    def __getitem__(self, item):
        if item not in self.plots:
            self.plots[item] = PlotContainer(self, item)

        return self.plots[item]

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')
            # eventually return True if all excetions are handled
        else:
            self._plot_all()
            return self

    def _plot_all(self):
        # def subplots(self, rows=2, figsize=(25, 10)):
        #     import matplotlib.pyplot as plt
        #     import matplotlib.dates as mdates
        #
        #     _, axes = plt.subplots(rows, 1,
        #                         sharex=True,
        #                         gridspec_kw={"height_ratios": [3, *([1] * (rows - 1))]},
        #                         figsize=figsize)
        #
        #     for ax in axes if isinstance(axes, Iterable) else [axes]:
        #         ax.xaxis_date()
        #         ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
        #
        #     return axes
        #
        # def plot(self, rows=2, cols=1, figsize=(18, 10), main_height_ratio=4):
        #     pass # return TaPlot(self.df, figsize, rows, cols, main_height_ratio)
        df = self.df
        hr = self.h_ratio
        wr = self.w_ratio
        mh = self.main_height
        rows = len(self.plots) + 1
        cols = 2 if self.plot_dist else 1

        height = mh  # + max(len(self.plots) - 1, 0) * (mh / sum(h_ratio) / h_ratio[1])
        grid_spec = {}

        if len(self.plots) > 1:
            grid_spec = {'height_ratios': [hr[0]] + [hr[1] for _ in range(1, len(self.plots))] + [1]}
        else:
            grid_spec = {'height_ratios': [hr[0], 1]}

        if self.plot_dist:
            grid_spec['width_ratios'] = wr

        # create subplots grid
        fig = plt.figure(figsize=(self.width, height), constrained_layout=True)
        gs = fig.add_gridspec(nrows=rows, ncols=cols, **grid_spec)
        sp = []

        for r in range(rows):
            if r == 0 or r >= rows - 1:
                shared_ax = [fig.add_subplot(gs[r, c]) for c in range(cols)]
                if isinstance(df.index, pd.DatetimeIndex):
                    shared_ax[0].xaxis_date()
                    shared_ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

                sp.append(shared_ax)
            else:
                sp.append([fig.add_subplot(gs[r, c], sharex=shared_ax[c]) if c == 0 else fig.add_subplot(gs[r, c]) for c in range(cols)])

        ax = np.array(sp).squeeze()
        self.fig = fig
        self.ax = ax

        # plot range slider
        range_ax = ax[-1, 0] if ax.ndim > 1 else ax[-1]
        range_ax.plot(df.index, df[self.range_slider_price].values)
        span_selector = SpanSelector(range_ax, onselect=self._plot_subplots, direction='horizontal',
                                     rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)

        self.widgets["selector"].append(span_selector)
        if self.cursor:
            self.widgets["cursor"].append(MultiCursor(fig.canvas, ax, horizOn=True, useblit=True, alpha=0.2))

        # initial span selection
        if self.subset is not None:
            sdf = df[self.subset].index
            xmin, xmax = sdf[0], sdf[-1]

            if isinstance(df.index, pd.DatetimeIndex):
                xmin, xmax = mdates.date2num(xmin), mdates.date2num(xmax)

            span_selector.stay_rect.set_bounds(xmin, 0, xmax - xmin, 1)
            span_selector.stay_rect.set_visible(True)
            span_selector.onselect(xmin, xmax)
        else:
            self._plot_subplots()

        # show figure now
        self.fig.show()

    def _plot_subplots(self, min_value=None, max_value=None):
        len_data = len(self.df.index) + 1
        start_idx = None
        stop_idx = None

        if min_value is not None:
            if np.abs(min_value - max_value) <= 2:
                min_value = None
                max_value = None
            else:
                if isinstance(self.df.index, pd.DatetimeIndex):
                    min_value = mdates.num2date(min_value).replace(tzinfo=None)

                for i, idx in enumerate(self.df.index):
                    if idx.replace(tzinfo=None) >= min_value:
                        start_idx = i
                        break

        if max_value is not None:
            if isinstance(self.df.index, pd.DatetimeIndex):
                max_value = mdates.num2date(max_value).replace(tzinfo=None)

            for i, idx in enumerate(reversed(self.df.index)):
                if idx.replace(tzinfo=None) <= max_value:
                    stop_idx = len_data - i
                    break

        for i in range(len(self.ax) - 1):
            ax = self.ax[i] if self.ax.ndim > 1 else [self.ax[i]]
            for a in ax: a.clear()

        keys = list(self.plots.keys())
        for a, p in self.plots.items():
            p.render(self.ax[keys.index(a)], slice(start_idx, stop_idx))

        if self.annotate:
            for c in self.widgets["data label"]: c.remove()
            self.widgets["data label"] = [mplcursors.cursor(ax) for ax in self.ax[:-1].flatten()]

        return min_value, max_value

    def __str__(self):
        return f'{self.width}/{self.main_height}'


if __name__ == '__main__':
    ## df = pd.DataFrame({"a": [1, 2, 3]})
    df = pd.read_csv("../pandas_ta_quant_plot_test/.data/SPY.csv", index_col="Date", parse_dates=True)
    print(df.tail())
    print(df.loc[['2019-11-11']][["Open", "Close"]])
    print(color_positive_negative(df).loc[['2019-11-25']])
    with df.ta_plot(range_slider_price='Close', backend=None) as p:
        p["main"].candlestick("Open", "High", "Low", "Close")
        p["main"].line(p.df["Close"])
        p["volume"].bar("Volume", colors=color_positive_negative(p.df))
        #p["macd"].plot(df["a"], "line", "line", "bar")
        #p > "dist"
    plt.show(block=True)