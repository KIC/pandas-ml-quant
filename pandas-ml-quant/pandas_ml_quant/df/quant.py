import pandas as pd

import pandas_ml_quant.indicators as indicators
import pandas_ml_quant.encoders as encoders
import pandas_ml_quant.labels as labels


class Quant(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def plot(self):
        pass


# add all indicators
def wrapper(func):
    def wrapped(quant, *args, **kwargs):
        return func(quant.df, *args, **kwargs)

    return wrapped


for indicator_functions in [indicators, encoders, labels]:
    for indicator_function in dir(indicator_functions):
        if indicator_function.startswith("ta_"):
            setattr(Quant, indicator_function, wrapper(getattr(indicator_functions, indicator_function)))



