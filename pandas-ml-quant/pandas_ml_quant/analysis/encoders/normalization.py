import numpy as _np
import pandas as _pd
from pandas_ml_common import Typing as _t
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


def ta_z_normalization(df: _t.PatchedPandas, period=90):
    # TODO imlpemnet a rolling z-scaling for all columns in the frame such that the have 0 mean and 1 stddev
    raise ValueError("not implemented yet")
    pass
