import numpy as np
from numba import guvectorize, float32, int32, float64, int64


@guvectorize([(float32[:], int32, float32[:]),
              (float64[:], int64, float64[:])], '(n),()->(n)')
def wilders_smoothing(arr: np.ndarray, period: int, res: np.ndarray):
    assert period > 0
    alpha = (period - 1) / period
    beta = 1 / period

    res[0:period] = np.nan
    res[period - 1] = arr[0:period].mean()
    for i in range(period, len(arr)):
        res[i] = alpha * res[i-1] + arr[i] * beta

