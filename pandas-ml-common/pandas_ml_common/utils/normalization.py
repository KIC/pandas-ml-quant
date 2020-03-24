
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

