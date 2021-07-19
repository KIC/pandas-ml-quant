"""Augment pandas DataFrame with methods for quant analysis plotting"""
__version__ = '0.2.7'

from collections import namedtuple
from pandas_ta_quant_plot.plots import *
from pandas.core.base import PandasObject
from pandas_ta_quant_plot.ta_plot_context import PlotContext

_ta = getattr(PandasObject, "ta", None)
if _ta is not None:
    if getattr(_ta, "plot", None) is None:
        setattr(PandasObject, "plot", lambda self, *args, **kwargs: PlotContext(self, *args, **kwargs))
else:
    ta = namedtuple("TA", ["plot"])
    setattr(PandasObject, "ta", lambda self, *args, **kwargs: ta(plot=PlotContext(self, *args, **kwargs)))

