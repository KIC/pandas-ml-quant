import pandas as _pd
import numpy as _np
# create convenient type hint
from pandas_ml_common import Typing as _t


def ta_decimal_year(df: _t.PatchedPandas):
    return (df.index.strftime("%j").astype(float) - 1) / 366 + df.index.strftime("%Y").astype(float)


def ta_sinusoidial_week_day(po: _t.PatchedPandas):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    return _np.sin(2 * _np.pi * (df.index.dayofweek / 6.0)).to_series(index=po.index, name="dow")


def ta_sinusoidial_week(po: _t.PatchedPandas):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    return _np.sin(2 * _np.pi * (df.index.week / 52.0)).to_series(index=po.index, name="week")
