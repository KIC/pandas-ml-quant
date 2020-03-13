import pandas as _pd

from pandas_ml_common import get_pandas_object as _get_pandas_object


def ta_candles_as_culb(df: _pd.DataFrame,  open="Open", high="High", low="Low", close="Close"):
    o = _get_pandas_object(df, open)
    c = _get_pandas_object(df, close)
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)
    oc = _pd.concat([o, c], axis=1)

    # calculate close, upper_shadow, lower_shadow, body
    return _pd.DataFrame({
        "close": c,
        "upper": (h - oc.max(axis=1)),
        "lower": (h - oc.min(axis=1)),
        "body": (c - o)
    }, index=df.index)
