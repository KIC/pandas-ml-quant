from functools import wraps

import pandas as pd
import logging
import pandas_ta_quant.technical_analysis as technichal_analysis

_log = logging.getLogger(__name__)


class TechnicalAnalysis(object):
    _ta_indicator_info = {}

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def help(self):
        return pd.DataFrame(TechnicalAnalysis._ta_indicator_info).T.rename(columns={0: 'module', 1: "doc"})


# add wrapper to call all indicators on data frames
def wrapper(func):
    @wraps(func)
    def wrapped(quant, *args, **kwargs):
        return func(quant.df, *args, **kwargs)

    return wrapped


# add indicators
for indicator_functions in [technichal_analysis]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            func = getattr(indicator_functions, indicator_function)
            setattr(TechnicalAnalysis, indicator_function[3:], wrapper(func))
            TechnicalAnalysis._ta_indicator_info[indicator_function] = [func.__module__, func.__doc__]


# add plotting
try:
    from pandas_ta_quant_plot import PlotContext
    setattr(TechnicalAnalysis, "plot", wrapper(PlotContext))
except Exception as e:
    _log.warning(f"pandas_ta_quant_plot not found: {e}")