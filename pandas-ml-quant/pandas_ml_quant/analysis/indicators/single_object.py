from typing import Union as _Union, Iterable

# create convenient type hint
import numpy as _np
import pandas as _pd
from scipy.stats import zscore

from pandas_ml_common import Typing
from pandas_ml_common.utils import has_indexed_columns
from pandas_ml_quant.analysis.filters import ta_ema as _ema, ta_wilders as _wilders, ta_sma as _sma
from pandas_ml_quant.analysis.utils import for_each_column
from pandas_ml_quant.utils import wilders_smoothing as _ws, with_column_suffix as _wcs
import pandas_ml_quant.analysis.bands as _bands

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_column
def ta_macd(df: Typing.PatchedPandas, fast_period=12, slow_period=26, signal_period=9, relative=True) -> _PANDAS:
    fast = _ema(df, fast_period)
    slow = _ema(df, slow_period)

    macd = (fast / slow - 1) if relative else (fast - slow)
    signal = _ema(macd, signal_period)
    hist = macd - signal
    suffix = f'{fast_period},{slow_period},{signal_period}'

    for label, frame in {f"macd_{suffix}": macd, f"signal_{suffix}": signal, f"histogram_{suffix}": hist}.items():
        frame.name = label

    macd = macd.to_frame() if isinstance(macd, _pd.Series) else macd
    return macd.join(signal).join(hist)


def ta_mom(df: _PANDAS, period=12, relative=True) -> _PANDAS:
    return _wcs(f"mom_{period}", df.pct_change(period) if relative else df.diff(period))


def ta_roc(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"roc_{period}", df.pct_change(period))


def ta_stddev(df: _PANDAS, period=12, nbdev=1, ddof=1, downscale=True) -> _PANDAS:
    return _wcs(f"stddev_{period}", (df.rolling(period).std(ddof=ddof) * nbdev) / (100 if downscale else 1))


def ta_rsi(df: _PANDAS, period=12):
    returns = df.diff()

    pos = _wilders(returns.clip(lower=0), period)
    neg = _wilders(_np.abs(returns.clip(upper=0)), period)

    return _wcs(f"rsi_{period}", pos / (pos + neg), returns)


def ta_apo(df: _PANDAS, fast_period=12, slow_period=26, exponential=False, relative=True) -> _PANDAS:
    fast = _ema(df, fast_period) if exponential else _sma(df, fast_period)
    slow = _ema(df, slow_period) if exponential else _sma(df, slow_period)
    apo = (fast / slow) if relative else (fast - slow)
    return _wcs(f"apo_{fast_period},{slow_period},{int(exponential)}", apo, df)


def ta_trix(df: _PANDAS, period=30) -> _PANDAS:
    return _wcs(f"trix_{period}", _ema(_ema(_ema(df, period), period), period).pct_change() * 100, df)


def ta_ppo(df: _pd.DataFrame, fast_period=12, slow_period=26, exponential=True) -> _PANDAS:
    fast = _ema(df, period=fast_period) if exponential else _sma(df, period=fast_period)
    slow = _ema(df, period=slow_period) if exponential else _sma(df, period=slow_period)
    return _wcs(f"ppo_{fast_period},{slow_period},{int(exponential)}", (fast - slow) / slow, df)


def ta_zscore(df: _PANDAS, period=20, ddof=1, downscale=True):
    res = df.rolling(period).apply(lambda c: zscore(c, ddof=ddof)[-1] / (4 if downscale else 1))

    return _wcs(f"z_{period}", res)


def ta_up_down_volatility_ratio(df: _PANDAS, period=60, normalize=True, setof_date=False):
    if isinstance(df, _pd.DataFrame):
        return df.apply(lambda col: ta_up_down_volatility_ratio(col, period, normalize))

    returns = df.pct_change() if normalize else df
    std = _pd.DataFrame({
        "std+": returns[returns > 0].rolling(period).std(),
        "std-": returns[returns < 0].rolling(period).std()
    }, index=returns.index).fillna(method="ffill")

    ratio = (std["std+"] / std["std-"] - 1).rename("std +/-")

    # eventually we can off set the date such that we can fake one continuous data frame
    if setof_date:
        # +7 -1 binds us approximately to the same week day
        ratio.index = ratio.index - _pd.DateOffset(days=(ratio.index[-1] - ratio.index[0]).days + 7 - 1)

    return ratio


@for_each_column
def ta_poly_coeff(df: _PANDAS, period=60, degree=2):
    from pandas_ml_common.utils import ReScaler
    from pandas_ml_common import np_nans
    from pandas_ml_common import inner_join

    x = _np.linspace(0, 1, period)
    v = df.values
    res = np_nans((len(df), degree + 1))

    for i in range(period, len(df)):
        y =  v[i - period:i]
        rescaler = ReScaler((y.min(), y.max()), (-1, 1))
        p, err, _, _, _ = _np.polyfit(x, rescaler(y), degree, full=True)
        res[i] = _np.hstack([p[:-1], err / period])

    return _pd.DataFrame(res, index=df.index, columns=[*[f'b{b}' for b in range(0, degree)], "mse"])


def ta_sharpe_ratio(df: _PANDAS, period=60, ddof=1, risk_free=0, is_returns=False):
    returns = df if is_returns else df.pct_change()
    mean_returns = returns.rolling(period).mean()
    sigma = returns.rolling(period).std(ddof=ddof)
    return (mean_returns - risk_free) / sigma


def ta_sortino_ratio(df: _PANDAS, period=60, ddof=1, risk_free=0, is_returns=False):
    returns = df if is_returns else df.pct_change()
    mean_returns = returns.rolling(period).mean()
    sigma = returns.clip(upper=risk_free).rolling(period).std(ddof=ddof)
    return (mean_returns - risk_free) / sigma


@for_each_column
def ta_draw_down(df: _PANDAS, return_dates=False, return_duration=False):
    ds = df
    pmin_pmax = (ds.diff(-1) > 0).astype(int).diff()  # <- -1 indicates pmin, +1 indicates pmax
    pmax = pmin_pmax[pmin_pmax == 1]
    pmin = pmin_pmax[pmin_pmax == -1]

    if pmin.index[0] < pmax.index[0]:
        pmin = pmin.drop(pmin.index[0])

    if pmin.index[-1] < pmax.index[-1]:
        pmax = pmax.drop(pmax.index[-1])

    dd = (_np.array(ds[pmin.index]) - _np.array(ds[pmax.index])) / _np.array(ds[pmax.index])
    d = {'drawdown': dd}

    if return_dates:
        d['d_start'] = pmax.index
        d['d_end'] = pmin.index

    if return_duration:
        dur = [_np.busday_count(p1.date(), p2.date()) for p1, p2 in zip(pmax.index, pmin.index)]
        d['duration'] = dur

    return _pd.DataFrame({}, index=df.index).join(_pd.DataFrame(d, index=pmax.index)).fillna(0)


def ta_ma_decompose(df: Typing.PatchedPandas, period=50, boost_ma=10):
    ma = _sma(df, period=period)
    base = ma.pct_change() * boost_ma
    ratio = df / ma - 1

    if isinstance(ratio, Typing.Series):
        ratio.name = f'{df.name}_ma_ratio'

    return base.to_frame().join(ratio)


def ta_ma_ratio(df: Typing.PatchedPandas, period=12, exponential='wilder') -> _PANDAS:
    if exponential is True:
        return (1 / _ema(df)) * df.values - 1
    if exponential == 'wilder':
        return (1 / _wilders(df)) * df.values - 1
    else:
        return (1 / _sma(df)) * df.values - 1


def ta_std_ret_bands_indicator(df: _pd.DataFrame, period=12, stddevs=[2.0], ddof=1, lag=1, scale_lag=True, include_mean=True, inculde_width=True) -> _PANDAS:
    columns = ["width"]
    res = _bands.ta_std_ret_bands(df, period, stddevs, ddof, lag, scale_lag, inculde_width=True)

    if isinstance(res.columns, _pd.MultiIndex):
        columns = [c for c in res.columns.to_list() if c[-1] in columns]

    return res[columns]


def ta_bbands_indicator(df: _PANDAS, period=12, stddev=2.0, ddof=1) -> _PANDAS:
    columns = ["width", "z-score", "quantile"]
    res = _bands.ta_bbands(df, period, stddev, ddof)

    if isinstance(res.columns, _pd.MultiIndex):
        columns = [c for c in res.columns.to_list() if c[-1] in columns]

    return res[columns]
