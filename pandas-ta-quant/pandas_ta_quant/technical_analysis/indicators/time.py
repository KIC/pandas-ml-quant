from datetime import date

import numpy as _np
import pandas as _pd
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas_ml_common import MlTypes as _t
from pandas_ta_quant._decorators import *
from pandas_ta_quant._utils import opex_date_of_month, opex_date_of_date


# create convenient type hint


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


@for_each_top_level_row
def ta_dist_opex(po: _t.PatchedPandas):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    def dist_next_opex(ts: pd.Timestamp):
        d = ts.date()
        opex = opex_date_of_date(d)
        if opex.day < d.day:
            next_opex = opex_date_of_date(d + relativedelta(months=1))
        else:
            next_opex = opex
            opex = opex_date_of_date(d - relativedelta(months=1))

        dist = (next_opex - d).days / (next_opex - opex).days
        return dist

    return df.index.to_series().apply(dist_next_opex).rename("dist_2_opex")

