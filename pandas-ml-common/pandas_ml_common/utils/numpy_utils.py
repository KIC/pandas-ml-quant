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


def empty_lists(shape):
    m = np.empty(shape, dtype=object)

    for i in np.ndindex(m.shape):
        m[i] = []

    return m


def get_buckets(arr, open=True, at_index=None):
    _arr = arr.squeeze()
    assert _arr.ndim == 1, f"Multi dimensional arrays not supported: got {arr.shape}"

    # build pairs
    if open:
        arr = np.empty(len(_arr) + 2)
        arr[1:len(_arr)+1] = _arr
        arr[[0, -1]] = np.nan
    else:
        arr = _arr

    tuples = [(arr[i], arr[i+1]) for i in range(len(arr)-1)]
    return tuples if at_index is None else tuples[at_index]

