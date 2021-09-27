import pandas as pd

from pandas_ml_common import MlTypes


class Forecast(object):

    def __init__(self, df: MlTypes.PatchedDataFrame, *args, **kwargs):
        self._df = df

    @property
    def df(self):
        return self._df

    def __str__(self):
        return str(self.df.groupby(level=0).tail(1)) if isinstance(self.df.index, pd.MultiIndex) else str(self.df.tail())

