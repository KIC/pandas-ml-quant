import numpy as np

from pandas_ml_common import Typing, np_nans
from pandas_ta_quant._decorators import *
from pandas_ta_quant._utils import with_column_suffix as _wcs, _rescale
from pandas_ta_quant.technical_analysis import filters as _f


@for_each_top_level_row
def ta_rescale(df: pd.DataFrame, range=(-1, 1), digits=None, axis=None):
    return _rescale(df, range, digits, axis)


@for_each_top_level_row
def ta_performance(df: Typing.PatchedPandas):
    return _wcs("performance", (1 + df.pct_change()).cumprod())


@for_each_top_level_row
def ta_returns(df: Typing.PatchedPandas, period=1):
    return _wcs("return", df.pct_change(periods=period))


@for_each_top_level_row
def ta_cumret(df: Typing.PatchedPandas, period=1):
    if period == 1:
        return (1 + df).cumprod()
    else:
        rets = np_nans((len(df), df.shape[1] if df.ndim > 1 else 1))
        rets[:period] = 1
        for i in range(period, len(df), period):
            rets[i] = (df.iloc[i] + 1) * rets[i-period]

        if df.ndim > 1:
            return pd.DataFrame(rets, columns=df.columns, index=df.index)
        else:
            return pd.Series(rets.squeeze(), name=df.name, index=df.index)


@for_each_top_level_row
def ta_log_returns(df: Typing.PatchedPandas, period=1):
    current = df
    lagged = df.shift(period)

    return _wcs("log_return", np.log(current) - np.log(lagged))


def ta_logret_as_return(df: Typing.PatchedPandas):
    return np.exp(df)


@for_each_top_level_row
def ta_cumlogret(df: Typing.PatchedPandas, period=1):
    return ta_cumret(ta_logret_as_return(df) - 1, period)


@for_each_top_level_row
def ta_ma_ratio(df: Typing.PatchedPandas, period=12, lag=0, ma='sma', **kwargs):
    mafunc = getattr(_f, f'ta_{ma}')
    return _wcs(f"{ma}({period}) x 1/", df / mafunc(df, period=period, **kwargs).shift(lag).values - 1, df)

# TODO reconstruct sma ratios