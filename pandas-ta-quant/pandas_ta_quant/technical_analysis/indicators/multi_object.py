from typing import Union as _Union

# create convenient type hint
import numpy as _np
import numpy as np
import pandas as _pd
import pandas as pd

import pandas_ta_quant.technical_analysis.filters as _f
from pandas_ml_common import get_pandas_object as _get_pandas_object
from pandas_ta_quant._decorators import *

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
@for_each_top_level_column
def ta_tr(df: _PANDAS, high="High", low="Low", close="Close", relative=True) -> _PANDAS:
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)
    c = _get_pandas_object(df, close).shift(1)

    if relative:
        ranges = (h / l - 1).rename("a").to_frame() \
            .join((h / c - 1).rename("b").abs()) \
            .join((l / c - 1).rename("c").abs())
    else:
        ranges = (h - l).rename("a").to_frame()\
            .join((h - c).rename("b").abs())\
            .join((l - c).rename("c").abs())

    return ranges.max(axis=1).rename("true_range")


@for_each_top_level_row
@for_each_top_level_column
def ta_atr(df: _PANDAS, period=14, high="High", low="Low", close="Close", relative=True, exponential='wilder') -> _PANDAS:
    if exponential is True:
        atr = _f.ta_ema(ta_tr(df, high, low, close, relative), period)
    if exponential == 'wilder':
        atr = _f.ta_wilders(ta_tr(df, high, low, close, relative), period)
    else:
        atr = _f.ta_sma(ta_tr(df, high, low, close, relative), period)

    return atr.rename(f"atr_{period}")


@for_each_top_level_row
@for_each_top_level_column
def ta_adx(df: _PANDAS, period=14, high="High", low="Low", close="Close", relative=True) -> _PANDAS:
    from pandas_ta_quant._utils import difference
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)

    temp = _pd.DataFrame({
        "up": difference(h, h.shift(1), relative, 0),
        "down": difference(l.shift(1), l, relative, 0)
    }, index=df.index)

    atr = ta_atr(df, period, high, low, close, relative=relative)
    pdm = _f.ta_wilders(temp.apply(lambda r: r[0] if r["up"] > r["down"] and r["up"] > 0 else 0, raw=False, axis=1), period)
    ndm = _f.ta_wilders(temp.apply(lambda r: r[1] if r["down"] > r["up"] and r["down"] > 0 else 0, raw=False, axis=1), period)

    pdi = pdm / atr
    ndi = ndm / atr
    adx = _f.ta_wilders((pdi - ndi).abs() / (pdi + ndi).abs(), period)

    return _pd.DataFrame({"+DM": pdm, "-DM": ndm, "+DI": pdi, "-DI": ndi, "ADX": adx}, index=df.index)


@for_each_top_level_row
@for_each_top_level_column
def ta_williams_R(df: _pd.DataFrame, period=14, close="Close", high="High", low="Low") -> _pd.Series:
    temp = _get_pandas_object(df, close).to_frame()
    temp = temp.join(_get_pandas_object(df, high if high is not None else close).rolling(period).max().rename("highest_high"))
    temp = temp.join(_get_pandas_object(df, low if low is not None else close).rolling(period).min().rename("lowest_low"))
    return ((temp["highest_high"] - temp[close]) / (temp["highest_high"] - temp["lowest_low"])).rename(f"williams_R_{period}")


@for_each_top_level_row
@for_each_top_level_column
def ta_ultimate_osc(df: _pd.DataFrame, period1=7, period2=14, period3=28, close="Close", high="High", low="Low") -> _pd.Series:
    # BP = Close - Minimum(Low or Prior Close).
    # TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    prev_close = _get_pandas_object(df, close).shift(1)
    downs = (_get_pandas_object(df, low if low is not None else close).to_frame().join(prev_close)).min(axis=1)
    ups = (_get_pandas_object(df, high if high is not None else close).to_frame().join(prev_close)).max(axis=1)
    temp = _pd.DataFrame({
        "bp": _get_pandas_object(df, close) - downs,
        "tr": ups - downs
    }, index=df.index)

    periods = [period1, period2, period3]
    avs = []
    for period in periods:
        # Average7 = (7 - period BP Sum) / (7 - period TR Sum)
        av = temp.rolling(period).sum()
        avs.append(av["bp"] / av["tr"])

    # UO = [(4 x Average7) + (2 x Average14) + Average28] / (4 + 2 + 1)
    return ((4 * avs[0] + 2 * avs[1] + avs[2]) / 7).rename(f"ultimate_osc_{period1},{period2},{period3}")


@for_each_top_level_row
@for_each_top_level_column
def ta_bop(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close") -> _PANDAS:
    # (CLOSE – OPEN) / (HIGH – LOW)
    o = _get_pandas_object(df, open)
    c = _get_pandas_object(df, close)
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)

    return ((c - o) / (h - l)).rename("bop")


@for_each_top_level_row
@for_each_top_level_column
def ta_cci(df: _pd.DataFrame, period=14, high="High", low="Low", close="Close", alpha=0.015) -> _PANDAS:
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)
    c = _get_pandas_object(df, close)

    tp = (h + l + c) / 3
    tp_sma = _f.ta_sma(tp, period)

    md = tp.rolling(period).apply(lambda x: _np.abs(x - x.mean()).sum() / period)
    return ((1 / alpha) * (tp - tp_sma) / md / 100).rename(f"cci_{period}")


@for_each_top_level_row
@for_each_top_level_column
def ta_gap(df: _pd.DataFrame, open="Open", close="Close") -> _PANDAS:
    return (df[open] / df[close].shift(1) - 1).rename("gap")


@for_each_top_level_row
@for_each_top_level_column
@is_time_consuming
def ta_vola_hurst(df: _PANDAS, period=255*2, lags=30, open="Open", high="High", low="Low", close="Close") -> _PANDAS:
    x = np.arange(1, lags)
    v = np.log(ta_gkyz_volatility(df, period=1, open=open, high=high, low=low, close=close)).rename("log_sqrt")

    def hurst(sig):
        # def del_Raw(s, q, x):
        #    return [np.mean(np.abs(s - s.shift(lag)) ** q) for lag in x]
        #
        # zeta_q = [np.polyfit(np.log(x), np.log(del_Raw(s, q, x)), 1)[0] for q in qVec]
        # h_est = np.polyfit(qVec, zeta_q, 1)[0]

        def dlsig2(sig, x):
            return [np.mean((sig - sig.shift(lag)) ** 2) for lag in x]

        model = np.polyfit(np.log(x), np.log(dlsig2(sig, x)), 1)
        return np.array([model[0] / 2., np.sqrt(np.exp(model[1]))])

    e_hurst = np.array([hurst(v.iloc[i-period+1:i]) for i in range(period - 1, len(df))])

    return df[[]].join(
        pd.DataFrame(e_hurst, columns=[f"H_{period}/{lags}", f"nu_{period}/{lags}"], index=df.index[period-1:]))


@for_each_top_level_row
@for_each_top_level_column
def ta_gkyz_volatility(df: _pd.DataFrame, period=12, open="Open", high="High", low="Low", close="Close") -> _PANDAS:
    # sqrt (sum of (  ln(o[i] / c[i-1] )^2 + 1/2 * (ln(h[i] / l[i]))^2 - (2 * ln(2) - 1) * (ln(c[i] / o[i]))^2   ) / N)
    vdf = df[[open, high, low, close]].copy()
    vdf.columns = ["o", "h", "l", "c"]
    vdf["c_1"] = df[close].shift(1)
    ln = np.log

    def one(r):
        return ln(r["o"] / r["c_1"])**2 + 0.5*(ln(r["h"] / r["l"]))**2 - (2 * ln(2) - 1)*(ln(r["c"] / r["o"]))**2

    v = np.sqrt(vdf.apply(one, axis=1).rolling(period).mean())
    return v.rename(f"gkyz_vol_{period}")


@for_each_top_level_row
@for_each_top_level_column
def ta_cc_volatility(df: _PANDAS, period=12, close="Close") -> _PANDAS:
    col = df if isinstance(df, pd.Series) else df[close]
    return np.sqrt((np.log(col / col.shift(1))**2).rolling(period).mean()).rename(f"cc_vol_{period}")




