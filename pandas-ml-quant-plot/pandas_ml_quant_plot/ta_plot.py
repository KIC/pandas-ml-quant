import matplotlib.pyplot as plt
from IPython.core.display import display
from matplotlib.widgets import RectangleSelector, SpanSelector
import logging
from pandas_ml_common import Typing
import pandas as pd


class PlotContext(object):

    def __init__(self,
                 df: Typing.PatchedDataFrame,
                 range_slider_price: str = "Close",
                 width: int = 20,
                 main_height: int = 11,
                 tail: int = 5 * 52,
                 backend='notebook'
                 ):
        self.old_backend = plt.get_backend()
        self.df = df
        self.range_slider_price = range_slider_price
        self.width = width
        self.main_height = main_height
        self.tail = tail
        self.target_backend = backend.replace('notebook', 'nbAgg')
        self.widgets = []

    def __enter__(self):
        self.old_backend = plt.get_backend()
        plt.switch_backend(self.target_backend)
        df = self.df

        range_fig, range_ax = plt.subplots(1, 1, figsize=(self.width, 2))
        range_ax.plot(df.index, df[self.range_slider_price].values)

        self.widgets.append(
            SpanSelector(range_ax, lambda *a, **kw: print(a, kw),
                         'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'), span_stays=True)
        )

        # range_fig.canvas.draw()
        plt.show()
        # bring back plotting:
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

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')
            # eventually return True if all excetions are handled

            plt.switch_backend(self.old_backend)

    def __str__(self):
        return f'{self.width}/{self.main_height} @ {self.target_backend}'
