from typing import Union as _Union

# create convenient type hint
import numpy as _np
import pandas as _pd

from pandas_ml_common.utils import has_indexed_columns
from pandas_ml_quant.utils import wilders_smoothing as _ws, with_column_suffix as _wcs

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_sma(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"sma_{period}", df.rolling(period).mean())


def ta_ema(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"ema_{period}", df.ewm(span=period, adjust=False, min_periods=period-1).mean())


def ta_wilders(df: _PANDAS, period=12) -> _PANDAS:
    if has_indexed_columns(df):
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


def ta_ewma_covariance(df: _PANDAS, convert_to='returns', alpha=0.97):
    data = df.copy()

    if convert_to == 'returns':
        data = df.pct_change()
    if convert_to == 'log-returns':
        data = _np.log(df) - _np.log(df.shift(1))

    data.columns = data.columns.to_list()
    return data.ewm(com=alpha).cov()


def ta_multi_bbands(s: _pd.Series, period=5, stddevs=[0.5, 1.0, 1.5, 2.0], ddof=1) -> _PANDAS:
    assert not has_indexed_columns(s)
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

