import pandas as pd

import pandas_ml_quant.indicators as indicators
import pandas_ml_quant.encoders as encoders
import pandas_ml_quant.labels as labels
from pandas_ml_quant.df.plot import TaPlot


class Quant(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def ta_plot(self, rows=2, cols=1, figsize=(18, 10)):
        return TaPlot(self.df, figsize, rows, cols)


# add wrapper to call all indicators on data frames
def wrapper(func):
    def wrapped(quant, *args, **kwargs):
        return func(quant.df, *args, **kwargs)

    return wrapped


# add indicators
for indicator_functions in [indicators, encoders, labels]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            setattr(Quant, indicator_function, wrapper(getattr(indicator_functions, indicator_function)))



