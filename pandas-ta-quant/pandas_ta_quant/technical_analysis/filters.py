from typing import Union as _Union

# create convenient type hint
import numpy as _np
import pandas as _pd

from pandas_ml_common import Typing as _t
from pandas_ml_common.utils import inner_join, add_multi_index
from pandas_ta_quant._decorators import *
from pandas_ta_quant._utils import wilders_smoothing as _ws, with_column_suffix as _wcs

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
def ta_sma(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"sma_{period}", df.rolling(period).mean())


@for_each_top_level_row
def ta_ema(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"ema_{period}", df.ewm(span=period, adjust=False, min_periods=period-1).mean())


@for_each_top_level_row
def ta_wilders(df: _PANDAS, period=12) -> _PANDAS:
    # NOTE this does not work with @for_each_column decorator!
    if df.ndim > 1:
        resdf = _pd.DataFrame({}, index=df.index)
        for col in df.columns:
            s = df[col].dropna()
            res = _np.zeros(s.shape)
            _ws(s.values, period, res)
            resdf = resdf.join(_pd.DataFrame({col: res}, index=s.index))

        res = resdf
    else:
        res = ta_wilders(df.to_frame(), period).iloc[:, 0]

    return _wcs(f"wilders_{period}", res)


@for_each_top_level_row
def ta_multi_bbands(s: _pd.Series, period=12, stddevs=[0.5, 1.0, 1.5, 2.0], ddof=1, include_mean=True) -> _PANDAS:
    assert s.ndim == 1, "Expected Series not Frame"
    mean = s.rolling(period).mean().rename("mean")
    std = s.rolling(period).std(ddof=ddof)
    df = _pd.DataFrame({}, index=mean.index)

    for stddev in reversed(stddevs):
        df[f'lower-{stddev}'] = mean - (std * stddev)

    if include_mean:
        df["mean"] = mean

    for stddev in stddevs:
        df[f'upper-{stddev}'] = mean + (std * stddev)

    return df


@for_each_top_level_row
def ta_multi_ma(df: _t.PatchedDataFrame, average_function='sma', period=12, factors=_np.linspace(1 - 0.2, 1 + 0.2, 5)) -> _t.PatchedDataFrame:
    ma = {'sma': ta_sma, 'ema': ta_ema, 'wilder': ta_wilders}
    res = _pd.DataFrame({}, index=df.index)

    if df.ndim > 1:
        res = None
        for col in df.columns.to_list():
            _df = ta_multi_ma(df[col], average_function, period, factors)
            _df = add_multi_index(_df, col)
            res = inner_join(res, _df, force_multi_index=True)

        return res

    for x in factors:
        res[f"{df.name}_{average_function}({period})({x:.3f})"] = ma[average_function](df, period) * x

    return res


