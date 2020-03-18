from typing import Union as _Union

import pandas as _pd
import numpy as _np

import pandas_ml_quant.indicators as _i
from pandas_ml_quant.utils import index_of_bucket as _index_of_bucket

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_future_crossings(df: _PANDAS, a=None, b=None, period=1, forecast=1):
    crossings = _i.ta_cross(df, a, b, period=period)

    if forecast > 1:
        crossings = _i.ta_rnn(crossings, range(1, forecast))

    return crossings.shift(-forecast)


def ta_future_bband_quantile(df: _pd.Series, forecast_period=5, period=14, stddev=2.0, ddof=1, include_mean=False):
    # we want to know if a future price is violating the current upper/lower band
    bands = _i.ta_bbands(df, period, stddev, ddof)
    bands = bands[["lower", "mean", "upper"] if include_mean else ["lower", "upper"] ]
    future = df.shift(-forecast_period)

    return bands \
        .join(future) \
        .apply(lambda row: _index_of_bucket(row[future.name], row[bands.columns]), axis=1, raw=False) \
        .rename(f"{df.name}_quantile")


def ta_opening_gap(df: _pd.DataFrame, offset=0.005, open="Open", close="Close"):
    gap = (df["Open"].shift(-1) / df["Close"]) - 1
    return gap.apply(lambda row: _np.nan if _np.isnan(row) or _np.isinf(row) else \
                                 2 if row > offset else 1 if row < -offset else 0)
