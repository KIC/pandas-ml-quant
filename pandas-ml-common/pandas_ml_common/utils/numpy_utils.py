import numpy as np


def nans(shape):
    arr = np.empty(shape)
    arr.fill(np.nan)
    return arr
