import numpy as _np
import pandas as _pd

from pandas_ml_common import get_pandas_object as _get_pandas_object


def ta_realative_candles(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume"):
    relative = _pd.DataFrame(index=df.index)
    relative[open] = (_np.log(df[open]) - _np.log(df[close].shift(1)))
    relative[close] = (_np.log(df[close]) - _np.log(df[close].shift(1)))
    relative[high] = (_np.log(df[high]) - _np.log(df[close].shift(1)))
    relative[low] = (_np.log(df[low]) - _np.log(df[close].shift(1)))

    if volume is not None:
        relative[volume] = relative[volume].pct_change()

    return relative


def ta_candles_as_culb(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume", relative_close=False):
    o = _get_pandas_object(df, open)
    c = _get_pandas_object(df, close)
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)
    oc = _pd.concat([o, c], axis=1)
    c_1 = c.shift(1)

    # calculate close, upper_shadow, lower_shadow, body
    res = _pd.DataFrame({
        "close": (c / c_1 - 1) if relative_close else c,
        "upper": (h / oc.max(axis=1) - 1),
        "lower": (oc.min(axis=1) / l - 1),
        "body": (c / o - 1)
    }, index=df.index)

    if volume is not None:
        res[volume] = df[volume].pct_change()

    return res