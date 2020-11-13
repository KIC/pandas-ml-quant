import pandas as pd

import pandas_ml_quant.technichal_analysis as technichal_analysis
import pandas_ml_quant.trading.strategy.optimized as optimized_strategies


# FIXME bring back plotting: from pandas_ml_quant.df_patching.plot import TaPlot


class TechnicalAnalysis(object):
    _ta_indicator_info = {}

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def help(self):
        return pd.DataFrame(TechnicalAnalysis._ta_indicator_info).T.rename(columns={0: 'module', 1: "doc"})

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


# add wrapper to call all indicators on data frames
def wrapper(func):
    def wrapped(quant, *args, **kwargs):
        return func(quant.df, *args, **kwargs)

    return wrapped


# add indicators
for indicator_functions in [technichal_analysis, optimized_strategies]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            func = getattr(indicator_functions, indicator_function)
            setattr(TechnicalAnalysis, indicator_function[3:], wrapper(func))
            TechnicalAnalysis._ta_indicator_info[indicator_function] = [func.__module__, func.__doc__]
