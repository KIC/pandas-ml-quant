from typing import Union, Tuple

import numpy as np
import scipy.stats as st

from pandas_ml_common import Typing


class NormalConfidence(object):

    def __init__(self, confidence: Union[float, Tuple[float, float]] = 0.95,):
        self.confidence = (
            np.abs(st.norm.ppf(confidence[0])),  # i.e 0.025
            np.abs(st.norm.ppf(confidence[1]))   # i.e 0.972
        ) if isinstance(confidence, tuple) else (
            np.abs(st.norm.ppf((1. - confidence) / 2.)),
        ) * 2

    def band(self, value, std=None):
        return self.lower(value, std), self.upper(value, std)

    def lower(self, value, std=None):
        value, std = self._extract_args(value, std)
        l = (value - std * self.confidence[0])
        return l.values.squeeze() if isinstance(l, Typing.AnyPandasObject) else l

    def upper(self, value, std=None):
        value, std = self._extract_args(value, std)
        u = (value + std * self.confidence[1])
        return u.values.squeeze() if isinstance(u, Typing.AnyPandasObject) else u

    def _extract_args(self, value, std):
        if std is None:
            if value.ndim >= 1:
                if isinstance(value, Typing.AnyPandasObject):
                    value = value._.values.squeeze()

                return value[0], value[1]
            else:
                raise ValueError("you need to pass a value, std tuple or value, std as arguments")
        else:
            return value, std

    def __call__(self, value, std=None):
        return self.band(value, std)