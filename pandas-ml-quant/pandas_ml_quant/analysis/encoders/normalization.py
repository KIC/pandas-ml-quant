import numpy as _np
import pandas as _pd

from pandas_ml_common.utils import ReScaler


def ta_rescale(df: _pd.DataFrame, range=(-1, 1), axis=None):
    if axis is not None:
        return df.apply(lambda x: ta_rescale(x, range), raw=False, axis=axis, result_type='broadcast')
    else:
        rescaler = ReScaler((df.values.min(), df.values.max()), range)
        rescaled = rescaler(df)

        if len(rescaled.shape) > 1:
            return _pd.DataFrame(rescaled, columns=df.columns, index=df.index)
        else:
            return _pd.Series(rescaled, name=df.name, index=df.index)


def ta_realative_candles(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close"):
    relative = _pd.DataFrame(index=df.index)
    relative[open] = (_np.log(df[open]) - _np.log(df[close].shift(1)))
    relative[close] = (_np.log(df[close]) - _np.log(df[close].shift(1)))
    relative[high] = (_np.log(df[high]) - _np.log(df[close].shift(1)))
    relative[low] = (_np.log(df[low]) - _np.log(df[close].shift(1)))
    return relative

