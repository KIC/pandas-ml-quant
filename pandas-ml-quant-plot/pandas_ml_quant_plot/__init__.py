"""Augment pandas DataFrame with methods for quant analysis plotting"""
__version__ = '0.2.0'

from pandas.core.base import PandasObject
from pandas_ml_quant_plot.ta_plot_context import PlotContext

setattr(PandasObject, "ta_plot", lambda self, *args, **kwargs: PlotContext(self, *args, **kwargs))


