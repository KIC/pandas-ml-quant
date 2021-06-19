import pandas as pd

from pandas_ml_common import Typing


class Forecast(object):

    def __init__(self, df: Typing.PatchedDataFrame, *args, **kwargs):
        self._df = df

    @property
    def df(self):
        return self._df

    def __str__(self):
        return str(self.df.groupby(level=0).tail(1)) if isinstance(self.df.index, pd.MultiIndex) else str(self.df.tail())

