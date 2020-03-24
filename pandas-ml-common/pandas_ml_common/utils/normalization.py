
from typing import Tuple as _Tuple

import numpy as _np
import pandas as _pd


class ReScaler(object):

    def __init__(self, domain: _Tuple[float, float], range: _Tuple[float, float]):
        self.domain = domain
        self.range = range
        self.rescale = _np.vectorize(self._rescale)

    def _interpolate(self, x: float):
        return self.range[0] * (1 - x) + self.range[1] * x

    def _uninterpolate(self, x: float):
        b = (self.domain[1] - self.domain[0]) if (self.domain[1] - self.domain[0]) != 0 else (1 / self.domain[1])
        return (x - self.domain[0]) / b

    def _rescale(self, x: float):
        return self._interpolate(self._uninterpolate(x))

    def __call__(self, *args, **kwargs):
        return self.rescale(args[0])


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

