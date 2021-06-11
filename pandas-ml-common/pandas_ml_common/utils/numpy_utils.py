from functools import partial
from typing import Iterable

import numpy as np


def one_hot(index, categories):
    if isinstance(index, Iterable):
        if categories is None: categories = index.max() + 1
        res = np.zeros((len(index), categories))
        for i, v in enumerate(index): res[i] = one_hot(v, categories)
        return res

    arr = np.zeros(categories)

    if np.issubdtype(np.uint32, np.integer):
        arr[index] = 1.0
    elif index.values >= 0:
        arr[index] = 1.0
    else:
        arr += np.NAN

    return arr


def clean_one_hot_classification(true_values: np.ndarray, predicted_values: np.ndarray):
    if isinstance(true_values, list):
        true_values = np.concatenate(true_values, axis=0)
        predicted_values = np.concatenate(predicted_values, axis=0)

    true_values = true_values.reshape(true_values.shape[0], -1)
    predicted_values = predicted_values.reshape(true_values.shape)

    # eventually convert integer values to one hot encoded values
    if true_values.ndim <= 1 < true_values.max():
        # convert integers to one hot
        true_values = one_hot(true_values, None)
    elif true_values.shape[-1] == 1:
        # fix binary classification case
        true_values = np.column_stack([1 - true_values, true_values])

    # fix binary classification case for labels
    if predicted_values.shape[-1] == 1:
        predicted_values = np.column_stack([1 - predicted_values, predicted_values])

    return true_values, predicted_values


def np_nans(shape):
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
    return WrappedTuple(tuples if at_index is None else tuples[at_index])


def mean(values):
    return np.array(values).mean()


class WrappedTuple(object):

    def __init__(self, tpl):
        self.tpl = tpl

    def as_tuple(self):
        return self.tpl

    def __getitem__(self, item):
        return self.tpl[item]

    def __repr__(self):
        return repr(self.tpl)

    def __str__(self):
        return str(self.tpl)


class CircularBuffer(object):

    def __init__(self, size, dtype=None):
        self.buffer = np_nans(size) if dtype is None else np_nans(size).astype(dtype)
        self.last_index = size - 1
        self.is_full = False
        self.i = -1

    def append(self, value):
        if self.i >= self.last_index:
            self.buffer = np.roll(self.buffer, -1)
            self.is_full = True
            self.i = 0
        else:
            self.i += 1

        self.buffer[self.i] = value

    def get_filled(self):
        return self.buffer if self.is_full else self.buffer[0:self.i+1]

    def __setitem__(self, key, value):
        self.buffer.__setitem__(key, value)

    def __getitem__(self, item):
        self.buffer.__getitem__(item)
