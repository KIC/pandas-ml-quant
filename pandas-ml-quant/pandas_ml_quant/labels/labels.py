import pandas as _pd
import numpy as _np
from typing import Union as _Union
import pandas_ml_quant.indicators as _i
from pandas_ml_common import get_pandas_object as _get_pandas_object

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


# FIXME
# FIXME                           OBSOLETE !!
# FIXME              functions either discrete or continuous !!!
# --------------------------------------------------------------
def ta_future_pct_to_current_mean(df: _pd.Series, forecast_period=1, period=14):
    future = df.shift(-forecast_period)
    mean = df.rolling(period).mean()

    return (future / mean) - 1


def ta_future_sma_cross(df: _pd.Series, forecast_period=14, fast_period=12, slow_period=26):
    assert isinstance(df, _pd.Series)
    fast = _i.ta_sma(df, fast_period)
    slow = _i.ta_sma(df, slow_period)
    cross = _i.ta_cross_over(None, fast, slow) | _i.ta_cross_under(None, fast, slow)
    return cross.shift(-forecast_period)


def ta_future_macd_cross(df: _pd.Series, forecast_period=14, fast_period=12, slow_period=26, signal_period=9):
    assert isinstance(df, _pd.Series)

    # calculate all macd crossings
    macd = _i.ta_macd(df, fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
    zero = macd["histogram"] * 0
    cross = _i.ta_cross_over(None, macd["histogram"], zero) | _i.ta_cross_under(None, macd["histogram"], zero)
    return cross.shift(-forecast_period)


def ta_future_bband_quantile(df: _pd.Series, forecast_period=14, period=5, stddev=2.0, ddof=1):
    # we want to know if a future price is violating the current upper/lower band
    bands = _i.ta_bbands(df, period, stddev, ddof)
    upper = bands["upper"]
    lower = bands["lower"]
    future = df.shift(-forecast_period)
    quantile = (future > upper).astype(int) - (future < lower).astype(int)

    return quantile


def ta_future_multiband_bucket(df: _pd.Series, forecast_period=14, period=5, stddevs=[0.5, 1.0, 1.5, 2.0], ddof=1):
    buckets = _i.ta_multi_bbands(df, period, stddevs=stddevs, ddof=ddof)
    future = df.shift(-forecast_period)

    # return index of bucket of which the future price lies in
    def index_of_bucket(value, data):
        if _np.isnan(value) or _np.isnan(data).any():
            return _np.nan

        for i, v in enumerate(data):
            if value < v:
                return i

        return len(data)

    return buckets \
        .join(future) \
        .apply(lambda row: index_of_bucket(row[future.name], row[buckets.columns]), axis=1, raw=False) \
        .rename(df.name)

