from typing import Union as _Union, Iterable

import numpy as _np
import pandas as _pd

from pandas_ta_quant._decorators import *

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
@for_each_column
def ta_std_ret_bands(s: _pd.Series, period=12, stddevs=[2.0], ddof=1, lag=1, scale_lag=True, include_mean=True, inculde_width=True) -> _PANDAS:
    if not isinstance(stddevs, Iterable):
        stddevs = [stddevs]

    mean = s.rolling(period).mean()
    std = s.pct_change().rolling(period).std(ddof=ddof).shift(lag) * (_np.sqrt(lag) if scale_lag else 1)
    res = mean.to_frame() if include_mean else _pd.DataFrame({}, index=s.index)

    for z in stddevs:
        res[f"upper_{z}"] = mean * (1 + std * z)
        res[f"lower_{z}"] = mean * (1 - std * z)

    if inculde_width:
        res["width"] = _np.log(res[f"upper_{stddevs[-1]}"] / res[f"lower_{stddevs[-1]}"]).rename("width")

    return res


@for_each_top_level_row
@for_each_column
def ta_bbands(df: _PANDAS, period=12, stddev=2.0, ddof=1, include_mean=True) -> _PANDAS:
    mean = df.rolling(period).mean().rename("mean")
    std = df.rolling(period).std(ddof=ddof)
    most_recent = df.rolling(period).apply(lambda x: x[-1], raw=True)

    upper = (mean + (std * stddev)).rename("upper")
    lower = (mean - (std * stddev)).rename("lower")
    z_score = ((most_recent - mean) / std).rename("z-score")
    width = _np.log(upper / lower).rename("width")
    quantile = _pd.Series(None, index=df.index, name='quantile', dtype='float64')
    quantile[most_recent <= lower] = 0
    quantile[most_recent > lower] = 1
    quantile[most_recent > mean] = 2
    quantile[most_recent > upper] = 3

    if include_mean:
        return _pd.DataFrame(upper) \
            .join(mean) \
            .join(lower) \
            .join(z_score) \
            .join(quantile) \
            .join(width) \
            .sort_index(axis=1)
    else:
        return _pd.DataFrame(upper) \
            .join(lower) \
            .join(z_score) \
            .join(quantile) \
            .join(width) \
            .sort_index(axis=1)
