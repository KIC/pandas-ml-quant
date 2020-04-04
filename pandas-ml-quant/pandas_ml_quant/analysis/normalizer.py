import numpy as np
import pandas as pd
from scipy.stats import norm

from pandas_ml_common import Typing, has_indexed_columns
from pandas_ml_quant.analysis import filters as _f
from pandas_ml_quant.utils import with_column_suffix as _wcs


def ta_returns(df: Typing.PatchedPandas):
    return _wcs("return", df.pct_change())


def ta_log_returns(df: Typing.PatchedPandas):
    current = df
    lagged = df.shift(1)

    return _wcs("log_return", np.log(current) - np.log(lagged))


def ta_ma_ratio(df: Typing.PatchedPandas, period=20, ma='sma', **kwargs):
    mafunc = getattr(_f, f'ta_{ma}')
    return _wcs(f"{ma}({period}) x 1/", df / mafunc(df, period=period, **kwargs) - 1, df)


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


def ta_z_norm(df: Typing.PatchedPandas, period=200, ddof=1, demean=True):
    if has_indexed_columns(df):
        return pd.DataFrame(
            {col: ta_z_norm(df[col], period, ddof, demean) for col in df.columns},
            index=df.index
        )

    # (value - mean) / std
    s = df.rolling(period).std()
    a = (df - df.rolling(period).mean()) if demean else df
    return (a / s / 4).rename(df.name)

