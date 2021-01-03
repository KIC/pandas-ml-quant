import pandas as pd
import numpy as np
from matplotlib.axis import Axis


class PlotContainer(object):

    def __init__(self, df):
        self.df = df

    def candlestick(self, *columns, **kwargs):
        pass

    def line(self, *columns, **kwargs):
        pass

    def bar(self, *columns, **kwargs):
        pass

    def plot(self, *columns, **kwargs):
        pass

    def render(self, ax: Axis):
        return ax

