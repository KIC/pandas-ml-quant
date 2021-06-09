from typing import Tuple

import numpy as np


class ECDF(object):

    def __init__(self, x):
        self.x = np.array(x, copy=True)
        self.x.sort()
        self.nobs = len(x)
        self.probs = np.linspace(1. / self.nobs, 1, self.nobs)

    def confidence_interval(self, lower: float, upper: float) -> Tuple[float, float]:
        # a[i-1] < v <= a[i]
        li = np.searchsorted(self.probs, lower, side='right')
        ri = np.searchsorted(self.probs, upper)

        li = [max(li - 1, 0), li]
        ri = [ri, min(ri + 1, self.nobs -1)]

        def get_val(idx, prob):
            weights = self.probs[idx] - prob
            vals = self.x[idx]
            return (vals * weights).sum() / weights.sum()

        return get_val(li, lower), get_val(ri, upper)

    def heat_bar(self, bins=21) -> Tuple[np.ndarray, np.ndarray]:
        # return a 2D array [value, mass]
        mass, edges = np.histogram(self.x, bins=bins, density=True)
        return edges[[0, -1]], mass
