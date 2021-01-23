from typing import Union as _Union

import numpy as _np
import pandas as _pd

import pandas_ta_quant.technical_analysis.bands as _b
import pandas_ta_quant.technical_analysis.filters as _f
import pandas_ta_quant.technical_analysis.indicators as _i
from pandas_ml_common import get_pandas_object as _get_pandas_object
from pandas_ta_quant._decorators import *
from pandas_ta_quant._utils import index_of_bucket as _index_of_bucket

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
def ta_cross_over(df: _pd.DataFrame, a=None, b=None, period=1) -> _PANDAS:
    # return only > 0
    return ta_cross(df, a, b, period).clip(lower=0)


@for_each_top_level_row
def ta_cross_under(df: _pd.DataFrame, a=None, b=None, period=1) -> _PANDAS:
    # return only < 0
    return ta_cross(df, a, b, period).clip(upper=0)


@for_each_top_level_row
def ta_cross(df: _pd.DataFrame, a=None, b=None, period=1):
    # get pandas objects for crossing
    if a is None and b is None:
        assert len(df.columns) == 2, f"ambiguous crossing of {df.columns}"
        a = df[df.columns[0]]
        b = df[df.columns[1]]
    elif b is None:
        b = _get_pandas_object(df, a)
        a = df
    elif a is None:
        b = _get_pandas_object(df, b)
        a = df
    else:
        a = _get_pandas_object(df, a)
        b = _get_pandas_object(df, b)

    # get periods
    a1 = a.shift(period)
    b1 = b.shift(period)

    # if a1 < b1 and a > b then a crosses over b
    a_over_b = ((a1 < b1) & (a > b)).astype(int)

    # if a1 > b1 and a < b then a cross under b
    a_under_b = ((a1 > b1) & (a < b)).astype(int) * -1

    return a_over_b + a_under_b


@for_each_top_level_row
def ta_future_crossings(df: _PANDAS, a=None, b=None, period=1, forecast=1):
    crossings = _i.ta_cross(df, a, b, period=period)

    if forecast > 1:
        crossings = _i.ta_rnn(crossings, range(1, forecast))

    return crossings.shift(-forecast)


@for_each_top_level_row
def ta_future_bband_quantile(df: _pd.Series, period=12, forecast_period=5, stddev=2.0, ddof=1, include_mean=True):
    # we want to know if a future price is violating the current upper/lower band
    bands = _b.ta_bbands(df, period, stddev, ddof)
    bands = bands[["lower", "mean", "upper"] if include_mean else ["lower", "upper"] ]
    future = df.shift(-forecast_period)

    return bands \
        .join(future) \
        .apply(lambda row: _index_of_bucket(row[future.name], row[bands.columns]), axis=1, raw=False) \
        .rename(f"{df.name}_quantile")


@for_each_top_level_row
def ta_future_multi_bband_quantile(df: _pd.Series, period=12, forecast_period=5, stddevs=[0.5, 1.0, 1.5, 2.0], ddof=1, include_mean=True):
    future = df.shift(-forecast_period)
    bands = _f.ta_multi_bbands(df, period, stddevs, ddof)

    if not include_mean:
        bands = bands.drop("mean", axis=1)

    return bands \
        .join(future) \
        .apply(lambda row: _index_of_bucket(row[future.name], row[bands.columns]), axis=1, raw=False) \
        .rename(f"{df.name}_quantile")


@for_each_top_level_row
def ta_future_multi_ma_quantile(df: _pd.Series, forecast_period=12, average_function='sma', period=12, factors=_np.linspace(1 - 0.2, 1 + 0.2, 5)):
    future = df.shift(-forecast_period)
    mas = _f.ta_multi_ma(df, average_function, period, factors)

    return mas \
        .join(future) \
        .apply(lambda row: _index_of_bucket(row[future.name], row[mas.columns]), axis=1, raw=False) \
        .rename(f"{df.name}_quantile")


@for_each_top_level_row
def ta_has_opening_gap(df: _pd.DataFrame, forecast_period=1, offset=0.005, open="Open", close="Close"):
    gap = (df[open].shift(-forecast_period) / df[close]) - 1
    return gap.apply(lambda row: _np.nan if _np.isnan(row) or _np.isinf(row) else \
                                 2 if row > offset else 1 if row < -offset else 0)\
              .rename("opening_gap")


@for_each_top_level_row
def ta_is_opening_gap_closed(df: _pd.DataFrame, threshold=0.001, no_gap=_np.nan, open="Open", high="High", low="Low", close="Close"):
    gap = "$gap$"
    yesterday_close = "$close-1$"
    df = df[[open, high, low, close]].copy()
    df[yesterday_close] = df[close].shift(1)
    df[gap] = (df[open] / df[close].shift(1)) - 1

    # if open > close-1 we need to check if low <= close-1
    # if open < close-1 we need to check if high >= close-1

    def gap_category(row):
        if row[gap] < threshold:
            return no_gap
        else:
            if row[open] > row[yesterday_close]:
                return row[low] <= row[yesterday_close]
            elif row[open] < row[yesterday_close]:
                return row[high] >= row[yesterday_close]
            else:
                return no_gap

    return df.apply(gap_category, raw=False, axis=1).rename("closing_gap")
