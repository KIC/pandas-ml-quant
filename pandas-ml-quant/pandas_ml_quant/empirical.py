from typing import Tuple

import numpy as np


class ECDF(object):

    def __init__(self, x):
        self.x = np.array(x, copy=True)
        self.x.sort()
        self.nobs = len(x)
        self.probs = np.linspace(1. / self.nobs, 1, self.nobs)

    def hist(self, bins='sqrt'):
        return np.histogram(self.x, bins=bins, density=True)

    def extreme(self):
        hist, edges = self.hist()
        iext = np.argmax(hist)
        return (edges[iext] + edges[iext + 1]) / 2

    def std(self):
        return np.std(self.x)

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

    def confidence_band_width(self, lower: float, upper: float) -> float:
        l, u = self.confidence_interval(lower, upper)
        return (u - l) / u

    def heat_bar(self, bins=21) -> Tuple[np.ndarray, np.ndarray]:
        # return a 2D array [value, mass]
        mass, edges = np.histogram(self.x, bins=bins, density=True)
        return edges[[0, -1]], mass

    def cvar(self, lower: float, upper: float) -> Tuple[float, float]:
        li = self.probs <= lower
        ri = self.probs >= upper
        return np.abs(self.x[li]).mean(), np.abs(self.x[ri]).mean()

    def get_tail_distance(self, observerd, lower: float, upper: float) -> Tuple[bool, bool]:
        left_tail = np.searchsorted(self.x, observerd, side='right')
        right_tail = np.searchsorted(self.x, observerd, side='left')
        left_prob = self.probs[min(left_tail, self.nobs - 1)]
        right_prob = self.probs[min(right_tail, self.nobs - 1)]
        return left_prob - lower, upper - right_prob

    def is_tail_event(self, observerd, lower: float, upper: float) -> Tuple[bool, bool]:
        left_tail_dist, right_tail_dist = self.get_tail_distance(observerd, lower, upper)
        return left_tail_dist < 0, right_tail_dist < 0

