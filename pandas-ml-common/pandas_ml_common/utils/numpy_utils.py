import numpy as np


def one_hot(index, len):
    arr = np.zeros(len)

    if np.issubdtype(np.uint32, np.integer):
        arr[index] = 1.0
    elif index.values >= 0:
        arr[index] = 1.0
    else:
        arr += np.NAN

    return arr


def nans(shape):
    arr = np.empty(shape)
    arr.fill(np.nan)
    return arr
