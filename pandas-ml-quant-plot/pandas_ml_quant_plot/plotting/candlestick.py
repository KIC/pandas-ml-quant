import matplotlib.dates as mdates
import pandas as pd
from mplfinance.original_flavor import candlestick_ohlc
from pandas_ml_common.plot.utils import new_fig_ts_axis

from pandas_ml_common.utils import get_pandas_object


def plot_candlestick(self, open="Open", high="High", low="Low", close="Close", ax=None, figsize=None, **kwargs):
    df = self._parent if hasattr(self, '_parent') else self

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