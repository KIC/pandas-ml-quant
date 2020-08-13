from time import sleep

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas_ml_common.plot.utils import matplot_dates
from .abstract_renderer import Renderer


class CandleStickRenderer(Renderer):

    def __init__(self, figsize=(20, 10)):
        super(CandleStickRenderer, self).__init__()
        matplotlib.use('Qt5Agg')
        plt.ion()

        self.fig, self.axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, figsize=figsize)
        self.fig.canvas.draw()  # draw and show it
        plt.show(block=False)

        for ax in self.axes:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

        self.r = 0

    def plot(self, old_state, action, new_state, reward, done):
        x = matplot_dates(new_state)
        o = new_state["Open"].values
        h = new_state["High"].values
        l = new_state["Low"].values
        c = new_state["Close"].values

        b = min(o, c)
        # if action was right other color then loosing action

        if reward > 0:
            color = 'black' if o > c else 'silver'
        else:
            color = 'red' if o > c else 'orange'

        self.r += reward

        self.axes[0].vlines(x, l, h, color=color)
        self.axes[0].bar(x, max(o, c) - b, bottom=b, color=color)
        self.axes[1].bar(x, self.r, color='silver')

    def render(self, mode=None):
        for ax in self.axes:
            ax.autoscale_view(tight=True, scalex=True, scaley=True)

        self.fig.canvas.draw()
        plt.pause(0.05)



