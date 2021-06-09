import pandas as pd
from pandas.core.base import PandasObject


def pandas_matplot_dates(df: PandasObject):
    import matplotlib.dates as mdates

    return mdates.date2num(df.index) if isinstance(df.index, pd.DatetimeIndex) else df.index
