from typing import Iterable

import pandas as pd

import pandas_ml_quant.analysis as analysis
import pandas_ml_quant.trading.strategy.optimized as optimized_strategies
from pandas_ml_common.utils import unique_level_rows, add_multi_index
from pandas_ml_quant.df.plot import TaPlot


class TechnicalAnalysis(object):
    info = {}

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def subplots(self, rows=2, figsize=(25, 10)):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        _, axes = plt.subplots(rows, 1,
                            sharex=True,
                            gridspec_kw={"height_ratios": [3, *([1] * (rows - 1))]},
                            figsize=figsize)

        for ax in axes if isinstance(axes, Iterable) else [axes]:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))

        return axes

    def plot(self, rows=2, cols=1, figsize=(18, 10), main_height_ratio=4):
        return TaPlot(self.df, figsize, rows, cols, main_height_ratio)


# add wrapper to call all indicators on data frames
def wrapper(func):
    def wrapped(quant, *args, **kwargs):
        if isinstance(quant.df.index, pd.MultiIndex):
            return pd.concat(
                [add_multi_index(func(quant.df.loc[group], *args, **kwargs), group, True, 0) for group in unique_level_rows(quant.df)],
                axis=0
            )

        return func(quant.df, *args, **kwargs)

    return wrapped


# add indicators
for indicator_functions in [analysis, optimized_strategies]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            func = getattr(indicator_functions, indicator_function)
            setattr(TechnicalAnalysis, indicator_function[3:], wrapper(func))
            TechnicalAnalysis.info[indicator_function] = func.__module__
            # TODO we also want to add func.__doc__ and return everything as a dataframe



