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

    def cvar(self, lower: float, upper: float) -> Tuple[float, float]:
        li = self.probs <= lower
        ri = self.probs >= upper
        return np.abs(self.x[li]).mean(), np.abs(self.x[ri]).mean()

    def is_tail_event(self, observerd, lower: float, upper: float) -> Tuple[bool, bool]:
        left_tail = np.searchsorted(self.x, observerd, side='right')
        right_tail = np.searchsorted(self.x, observerd, side='left')
        left_prob = self.probs[min(left_tail, self.nobs - 1)]
        right_prob = self.probs[min(left_tail, self.nobs - 1)]
        return left_prob <= lower, right_prob >= upper

