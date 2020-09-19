
from typing import Tuple as _Tuple

import numpy as _np
import pandas as _pd

from pandas_ml_common.utils import unpack_nested_arrays


class ReScaler(object):

    def __init__(self, domain: _Tuple[float, float], range: _Tuple[float, float], clip=False):
        self.domain = domain
        self.range = range
        self.rescale = _np.vectorize(self._rescale)
        self.clip = clip

    def _interpolate(self, x: float):
        return self.range[0] * (1 - x) + self.range[1] * x

    def _uninterpolate(self, x: float):
        b = (self.domain[1] - self.domain[0]) if (self.domain[1] - self.domain[0]) != 0 else (1 / self.domain[1])
        return (x - self.domain[0]) / b

    def _rescale(self, x: float):
        return self._interpolate(self._uninterpolate(x))

    def __call__(self, *args, **kwargs):
        if not self.clip:
            return self.rescale(args[0])
        else:
            if self.range[0] < self.range[1]:
                return _np.clip(self.rescale(args[0]), self.range[0], self.range[1])
            else:
                return _np.clip(self.rescale(args[0]), self.range[1], self.range[0])


def ecdf(v):
    if isinstance(v, (_pd.DataFrame, _pd.Series)):
        v = unpack_nested_arrays(v)

    shape = v.shape
    x = v.flatten()
    x = _np.sort(x)
    return ((_np.searchsorted(x, v, side='right') + 1) / len(v)).reshape(shape)
