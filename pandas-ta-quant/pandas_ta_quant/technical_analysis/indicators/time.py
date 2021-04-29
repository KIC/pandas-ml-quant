import numpy as _np
import pandas as _pd

# create convenient type hint
from pandas_ml_common import Typing as _t
from pandas_ta_quant._decorators import *


@for_each_top_level_row
def ta_decimal_year(df: _t.PatchedPandas):
    return ((df.index.strftime("%j").astype(float) - 1) / 366 + df.index.strftime("%Y").astype(float))\
        .to_series(index=df.index, name="decimal_time")


@for_each_top_level_row
def ta_sinusoidal_week_day(po: _t.PatchedPandas):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    return _np.sin(2 * _np.pi * (df.index.dayofweek / 6.0)).to_series(index=po.index, name="dow")


@for_each_top_level_row
def ta_sinusoidal_week(po: _t.PatchedPandas):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    return _np.sin(2 * _np.pi * (df.index.isocalendar().week / 52.0)).rename("week")
