from time import sleep
from typing import Dict, Tuple, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas_ml_common.plot.utils import matplot_dates
from .abstract_renderer import Renderer


class CandleStickRenderer(Renderer):

    def __init__(self, action_mapping: List[Tuple[int, str, str]] = None, figsize=(20, 10)):
        super(CandleStickRenderer, self).__init__()
        matplotlib.use('Qt5Agg')
        plt.ion()

        self.action_map = action_mapping
        self.figsize = figsize

        self.min_time_step = 1.0
        self.r = 0

        self.reset()

    def reset(self):
        self.r = 0

        self.fig, self.axes = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}, figsize=self.figsize)
        self.fig.canvas.draw()  # draw and show it
        plt.show(block=False)

        for ax in self.axes:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

    def plot(self, old_state, action, new_state, reward, done):
        df = new_state.to_frame().T
        x = matplot_dates(df)
        o = df["Open"].values
        h = df["High"].values
        l = df["Low"].values
        c = df["Close"].values

        # indicator x is half a time step before / after the trading day
        bx = x - (self.min_time_step * 0.4)
        sx = x + (self.min_time_step * 0.4)

        # plot candle
        b = min(o, c)
        color = 'red' if o > c else 'green'
        self.axes[0].vlines(x, l, h, color=color)
        self.axes[0].bar(x, max(o, c) - b, bottom=b, color=color)

        # plot actions
        for _action, plot_act, column in self.action_map:
            if _action == action:
                if plot_act == 'buy':
                    self.axes[0].scatter(bx, df[column], marker=">", color='green')
                if plot_act == 'sell':
                    self.axes[0].scatter(sx, df[column], marker="<", color='red')

        # plot reward
        self.r += reward.item() if isinstance(reward, np.ndarray) else float(reward)
        print(x, self.r)
        self.axes[1].bar(x, self.r, color='silver')

        if done:
            sleep(1)
            plt.close(self.fig)
            self.reset()

    def render(self, mode=None, min_time_step=1.0):
        self.min_time_step = min_time_step

        for ax in self.axes:
            ax.autoscale_view(tight=True, scalex=True, scaley=True)

        self.fig.canvas.draw()
        plt.pause(0.05)



