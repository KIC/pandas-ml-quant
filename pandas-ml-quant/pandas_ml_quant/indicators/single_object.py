from typing import Union as _Union

# create convenient type hint
import numpy as _np
import pandas as _pd
from scipy.stats import zscore

from pandas_ml_quant.utils import wilders_smoothing as _ws, with_column_suffix as _wcs

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_inverse(df: _PANDAS) -> _PANDAS:
    return df.apply(lambda col: col * -1 + col.min() + col.max(), raw=True)


def ta_sma(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"sma_{period}", df.rolling(period).mean())


def ta_ema(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"ema_{period}", df.ewm(span=period, adjust=False, min_periods=period-1).mean())


def ta_wilders(df: _PANDAS, period=12) -> _PANDAS:
    if isinstance(df, _pd.Series):
        res = ta_wilders(df.to_frame(), period).iloc[:, 0]
    else:
        resdf = _pd.DataFrame({}, index=df.index)
        for col in df.columns:
            s = df[col].dropna()
            res = _np.zeros(s.shape)
            _ws(s.values, period, res)
            resdf = resdf.join(_pd.DataFrame({col: res}, index=s.index))

        res = resdf

    return _wcs(f"wilders_{period}", res)


def ta_macd(df: _PANDAS, fast_period=12, slow_period=26, signal_period=9, relative=True) -> _PANDAS:
    fast = ta_ema(df, fast_period)
    slow = ta_ema(df, slow_period)
    macd = (fast / slow - 1) if relative else (fast - slow)
    signal = ta_ema(macd, signal_period)
    hist = macd - signal
    suffix = f'{fast_period},{slow_period},{signal_period}'

    for label, frame in {f"macd_{suffix}": macd, f"signal_{suffix}": signal, f"histogram_{suffix}": hist}.items():
        if isinstance(frame, _pd.DataFrame) and len(df.columns) > 1:
            frame.columns = _pd.MultiIndex.from_product([frame.columns, [label]])
        else:
            frame.name = label

    macd = macd.to_frame() if isinstance(macd, _pd.Series) else macd
    return macd.join(signal).join(hist)


def ta_mom(df: _PANDAS, period=10) -> _PANDAS:
    return _wcs(f"mom_{period}", df.diff(period))


def ta_roc(df: _PANDAS, period=10) -> _PANDAS:
    return _wcs(f"roc_{period}", df.pct_change(period))


def ta_stddev(df: _PANDAS, period=5, nbdev=1, ddof=1) -> _PANDAS:
    return _wcs(f"stddev_{period}", (df.rolling(period).std(ddof=ddof) * nbdev))


def ta_rsi(df: _PANDAS, period=14):
    returns = df.diff()

    pos = ta_wilders(returns.clip(lower=0), period)
    neg = ta_wilders(_np.abs(returns.clip(upper=0)), period)

    return _wcs(f"rsi_{period}", pos / (pos + neg), returns)


def ta_apo(df: _PANDAS, fast_period=12, slow_period=26, exponential=False, relative=True) -> _PANDAS:
    fast = ta_ema(df, fast_period) if exponential else ta_sma(df, fast_period)
    slow = ta_ema(df, slow_period) if exponential else ta_sma(df, slow_period)
    apo = (fast / slow) if relative else (fast - slow)
    return _wcs(f"apo_{fast_period},{slow_period},{int(exponential)}", apo, df)


def ta_trix(df: _PANDAS, period=30) -> _PANDAS:
    return _wcs(f"trix_{period}", ta_ema(ta_ema(ta_ema(df, period), period), period).pct_change() * 100, df)


def ta_ppo(df: _pd.DataFrame, fast_period=12, slow_period=26, exponential=True) -> _PANDAS:
    fast = ta_ema(df, period=fast_period) if exponential else ta_sma(df, period=fast_period)
    slow = ta_ema(df, period=slow_period) if exponential else ta_sma(df, period=slow_period)
    return _wcs(f"ppo_{fast_period},{slow_period},{int(exponential)}", (fast - slow) / slow, df)


def ta_zscore(df: _PANDAS, period=20, ddof=1):
    res = df.rolling(period).apply(lambda c: zscore(c, ddof=ddof)[-1])

    return _wcs(f"z_{period}", res)


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


def ta_ewma_covariance(df: _PANDAS, convert_to='returns', alpha=0.97):
    data = df

    if convert_to == 'returns':
        data = df.pct_change()
    if convert_to == 'log-returns':
        data = _np.log(df) - _np.log(df.shift(1))

    return data.ewm(com=alpha).cov()


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


def ta_multi_bbands(s: _pd.Series, period=5, stddevs=[0.5, 1.0, 1.5, 2.0], ddof=1) -> _PANDAS:
    assert isinstance(s, _pd.Series)
    mean = s.rolling(period).mean().rename("mean")
    std = s.rolling(period).std(ddof=ddof)
    df = _pd.DataFrame({}, index=mean.index)

    for stddev in reversed(stddevs):
        df[f'lower-{stddev}'] = mean - (std * stddev)

    df["mean"] = mean

    for stddev in stddevs:
        df[f'upper-{stddev}'] = mean + (std * stddev)

    return df


def ta_bbands(df: _PANDAS, period=5, stddev=2.0, ddof=1) -> _PANDAS:
    mean = df.rolling(period).mean()
    std = df.rolling(period).std(ddof=ddof)
    most_recent = df.rolling(period).apply(lambda x: x[-1], raw=True)

    upper = mean + (std * stddev)
    lower = mean - (std * stddev)
    z_score = (most_recent - mean) / std
    quantile = (most_recent > upper).astype(int) - (most_recent < lower).astype(int)

    if isinstance(mean, _pd.Series):
        upper.name = "upper"
        mean.name = "mean"
        lower.name = "lower"
        z_score.name = "z"
        quantile.name = "quantile"
    else:
        upper.columns = _pd.MultiIndex.from_product([upper.columns, ["upper"]])
        mean.columns = _pd.MultiIndex.from_product([mean.columns, ["mean"]])
        lower.columns = _pd.MultiIndex.from_product([lower.columns, ["lower"]])
        z_score.columns = _pd.MultiIndex.from_product([z_score.columns, ["z"]])
        quantile.columns = _pd.MultiIndex.from_product([z_score.columns, ["quantile"]])

    return _pd.DataFrame(upper) \
        .join(mean) \
        .join(lower) \
        .join(z_score) \
        .join(quantile) \
        .sort_index(axis=1)
