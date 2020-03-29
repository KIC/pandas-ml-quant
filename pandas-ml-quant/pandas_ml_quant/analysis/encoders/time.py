from typing import Union as _Union

# create convenient type hint
import pandas as _pd

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_decimal_year(df: _PANDAS):
    return df.index.strftime("%j").astype(float) - 1 / 366 + df.index.strftime("%Y").astype(float)


def ta_week_day(po: _PANDAS):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    return (df.index.dayofweek / 6.0).to_series(index=po.index, name="dow")


def ta_week(po: _PANDAS):
    if not isinstance(po.index, _pd.DatetimeIndex):
        df = po.copy()
        df.index = _pd.to_datetime(df.index)
    else:
        df = po

    return (df.index.week / 52.0).to_series(index=po.index, name="week")
