import numpy as np
import pandas as pd
from scipy.stats import norm

from pandas_ml_common import Typing, has_indexed_columns
from pandas_ml_common.utils import ReScaler
from pandas_ml_quant.analysis import filters as _f
from pandas_ml_quant.utils import with_column_suffix as _wcs


def ta_rescale(df: pd.DataFrame, range=(-1, 1), digits=None, axis=None):
    if axis is not None:
        return df.apply(lambda x: ta_rescale(x, range), raw=False, axis=axis, result_type='broadcast')
    else:
        rescaler = ReScaler((df.values.min(), df.values.max()), range)
        rescaled = rescaler(df.values)

        if digits is not None:
            rescaled = np.around(rescaled, digits)

        if rescaled.ndim > 1:
            return pd.DataFrame(rescaled, columns=df.columns, index=df.index)
        else:
            return pd.Series(rescaled, name=df.name, index=df.index)


def ta_returns(df: Typing.PatchedPandas):
    return _wcs("return", df.pct_change())


def ta_log_returns(df: Typing.PatchedPandas):
    current = df
    lagged = df.shift(1)

    return _wcs("log_return", np.log(current) - np.log(lagged))


def ta_ma_ratio(df: Typing.PatchedPandas, period=12, lag=0, ma='sma', **kwargs):
    mafunc = getattr(_f, f'ta_{ma}')
    return _wcs(f"{ma}({period}) x 1/", df / mafunc(df, period=period, **kwargs).shift(lag).values - 1, df)


def ta_ncdf_compress(df: Typing.PatchedPandas, period=200, lower_percentile=25, upper_percentile=75) -> Typing.PatchedPandas:
    if has_indexed_columns(df):
        return pd.DataFrame(
            {col: ta_ncdf_compress(df[col], period, lower_percentile, upper_percentile) for col in df.columns},
            index=df.index
        )

    f50 = df.rolling(period).mean().rename("f50")
    fup = df.rolling(period).apply(lambda x: np.percentile(x, upper_percentile)).rename("fup")
    flo = df.rolling(period).apply(lambda x: np.percentile(x, lower_percentile)).rename("flo")

    return pd.Series(norm.cdf(0.5 * (df - f50) / (fup - flo)) - 0.5, index=df.index, name=df.name)


def ta_z_norm(df: Typing.PatchedPandas, period=200, ddof=1, demean=True, lag=0):
    if has_indexed_columns(df):
        return pd.DataFrame(
            {col: ta_z_norm(df[col], period, ddof, demean) for col in df.columns},
            index=df.index
        )

    # (value - mean) / std
    s = df.rolling(period).std(ddof=ddof)
    a = (df - df.rolling(period).mean().shift(lag)) if demean else df
    return (a / s / 4).rename(df.name)


def ta_performance(df: Typing.PatchedPandas):
    delta = df.pct_change() + 1
    return delta.cumprod()


def _ta_adaptive_normalisation():
    # TODO implement .papers/Adaptive Normalization.pdf
    pass