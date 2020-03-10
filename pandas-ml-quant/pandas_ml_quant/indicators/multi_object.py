from typing import Union as _Union

# create convenient type hint
import numpy as _np
import pandas as _pd
from pandas.core.base import PandasObject

from pandas_ml_quant.indicators.single_object import ta_ema, ta_wilders, ta_sma
from pandas_ml_common.df.utils.index_utils import get_pandas_object

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_tr(df: _PANDAS, high="High", low="Low", close="Close", relative=False) -> _PANDAS:
    h = get_pandas_object(df, high)
    l = get_pandas_object(df, low)
    c = get_pandas_object(df, close).shift(1)

    if relative:
        ranges = (h / l - 1).rename("a").to_frame() \
            .join((h / c - 1).rename("b").abs()) \
            .join((l / c - 1).rename("c").abs())
    else:
        ranges = (h - l).rename("a").to_frame()\
            .join((h - c).rename("b").abs())\
            .join((l - c).rename("c").abs())

    return ranges.max(axis=1).rename("true_range")


def ta_atr(df: _PANDAS, period=14, high="High", low="Low", close="Close", relative=False, exponential='wilder') -> _PANDAS:
    if exponential is True:
        return ta_ema(ta_tr(df, high, low, close, relative), period)
    if exponential == 'wilder':
        return ta_wilders(ta_tr(df, high, low, close, relative), period)
    else:
        return ta_sma(ta_tr(df, high, low, close, relative), period)


def ta_adx(df: _PANDAS, period=14, high="High", low="Low", close="Close", relative=True) -> _PANDAS:
    h = get_pandas_object(df, high)
    l = get_pandas_object(df, low)

    temp = _pd.DataFrame({
        "up": (h / h.shift(1) - 1) if relative else (h - h.shift(1)),
        "down": (l.shift(1) / l - 1) if relative else (l.shift(1) - l)
    }, index=df.index)

    atr = ta_atr(df, period, high, low, close, relative=False)
    pdm = ta_wilders(temp.apply(lambda r: r[0] if r["up"] > r["down"] and r["up"] > 0 else 0, raw=False, axis=1), period)
    ndm = ta_wilders(temp.apply(lambda r: r[1] if r["down"] > r["up"] and r["down"] > 0 else 0, raw=False, axis=1), period)

    pdi = pdm / atr
    ndi = ndm / atr
    adx = ta_wilders((pdi - ndi).abs() / (pdi + ndi).abs(), period)

    return _pd.DataFrame({"+DM": pdm, "-DM": ndm, "+DI": pdi, "-DI": ndi, "ADX": adx}, index=df.index)


def ta_williams_R(df: _pd.DataFrame, period=14, close="Close", high="High", low="Low") -> _pd.Series:
    temp = get_pandas_object(df, close).to_frame()
    temp = temp.join(get_pandas_object(df, high if high is not None else close).rolling(period).max().rename("highest_high"))
    temp = temp.join(get_pandas_object(df, low if low is not None else close).rolling(period).min().rename("lowest_low"))
    return (temp["highest_high"] - temp[close]) / (temp["highest_high"] - temp["lowest_low"])


def ta_ultimate_osc(df: _pd.DataFrame, period1=7, period2=14, period3=28, close="Close", high="High", low="Low") -> _pd.Series:
    # BP = Close - Minimum(Low or Prior Close).
    # TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    prev_close = get_pandas_object(df, close).shift(1)
    downs = (get_pandas_object(df, low if low is not None else close).to_frame().join(prev_close)).min(axis=1)
    ups = (get_pandas_object(df, high if high is not None else close).to_frame().join(prev_close)).max(axis=1)
    temp = _pd.DataFrame({
        "bp": get_pandas_object(df, close) - downs,
        "tr": ups - downs
    }, index=df.index)

    periods = [period1, period2, period3]
    avs = []
    for period in periods:
        # Average7 = (7 - period BP Sum) / (7 - period TR Sum)
        av = temp.rolling(period).sum()
        avs.append(av["bp"] / av["tr"])

    # UO = [(4 x Average7) + (2 x Average14) + Average28] / (4 + 2 + 1)
    return (4 * avs[0] + 2 * avs[1] + avs[2]) / 7


def ta_bop(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close") -> _PANDAS:
    # (CLOSE – OPEN) / (HIGH – LOW)
    o = get_pandas_object(df, open)
    c = get_pandas_object(df, close)
    h = get_pandas_object(df, high)
    l = get_pandas_object(df, low)

    return (c - o) / (h - l)


def ta_cci(df: _pd.DataFrame, period=14, high="High", low="Low", close="Close", alpha=0.015) -> _PANDAS:
    h = get_pandas_object(df, high)
    l = get_pandas_object(df, low)
    c = get_pandas_object(df, close)

    tp = (h + l + c) / 3
    tp_sma = ta_sma(tp, period)
    md = tp.rolling(period).apply(lambda x: _np.abs(x - x.mean()).sum() / period)
    return (1 / alpha) * (tp - tp_sma) / md / 100

# TODO use get_pandas_object in all the cases from here!
def ta_cross_over(df: _pd.DataFrame, a, b=None, period=1) -> _PANDAS:
    if isinstance(a, int):
        if isinstance(df, _pd.Series):
            a = _pd.Series(_np.ones(len(df)) * a, name=df.name, index=df.index)
        else:
            a = _pd.DataFrame({c: _np.ones(len(df)) * a for c in df.columns}, index=df.index)

    if b is None:
        b = a
        a = df

    old_a = (a if isinstance(a, PandasObject) else df[a]).shift(period)
    young_a = (a if isinstance(a, PandasObject) else df[a])

    if isinstance(b, int):
        if isinstance(old_a, _pd.Series):
            b = _pd.Series(_np.ones(len(df)) * b, name=old_a.name, index=old_a.index)
        else:
            b = _pd.DataFrame({c : _np.ones(len(df)) * b for c in old_a.columns}, index=old_a.index)

    old_b = (b if isinstance(b, PandasObject) else df[b]).shift(period)
    young_b = (b if isinstance(b, PandasObject) else df[b])

    return (old_a <= old_b) & (young_a > young_b)


def ta_cross_under(df: _pd.DataFrame, a, b=None, period=1) -> _PANDAS:
    if b is None:
        b = a
        a = df

    return ta_cross_over(df, b, a, period)

