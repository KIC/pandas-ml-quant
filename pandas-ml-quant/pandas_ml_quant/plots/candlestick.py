import matplotlib.dates as mdates
import pandas as pd
from mpl_finance import candlestick_ohlc

from pandas_ml_common import get_pandas_object
from pandas_ml_quant.plots.utils import new_fig_ts_axis


def ta_candlestick(self, open="Open", high="High", low="Low", close="Close", ax=None, figsize=None, **kwargs):
    df = self if isinstance(self, pd.DataFrame) else self._parent

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    # Plot candlestick chart
    data = pd.DataFrame({
        "Date": mdates.date2num(df.index),
        "open": get_pandas_object(df, open),
        "high": get_pandas_object(df, high),
        "low": get_pandas_object(df, low),
        "close": get_pandas_object(df, close),
    })

    candlestick_ohlc(ax, data.values, width=0.6, colorup='g', colordown='r')
    return ax