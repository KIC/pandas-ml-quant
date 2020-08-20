from typing import Tuple

import numpy as np

from .abstract_buffer import Buffer


class ArrayBuffer(Buffer):

    def __init__(self, shape: Tuple[int], dtype='float'):
        self.buffer = np.empty(shape, dtype=dtype)
        self.size = shape[0]
        self.cnt = 0

    def add(self, row: np.ndarray) -> 'ArrayBuffer':
        assert row.shape == self.buffer.shape[1:]
        self.buffer = np.roll(self.buffer, -1, 0)
        self.buffer[-1] = row
        self.cnt += 1
        return self

    @property
    def data(self) -> np.ndarray:
        return self.buffer

    @property
    def is_full(self):
        return self.cnt >= self.size

    def valid_buffer(self):
        return self if self.is_full else self._copy(self.buffer[-self.cnt-1:])

    def sort_by_column(self, col: int = 0) -> 'ArrayBuffer':
        buffer = self._copy(self.buffer[np.argsort(self.buffer[:, col])])
        return buffer

    def rank_and_cut(self, percentile: Tuple[int, float]) -> 'ArrayBuffer':
        column, percentile = percentile
        sorted_buffer = self.sort_by_column(column)
        index = int(percentile * (sorted_buffer.size / 100) * 100)
        return self._copy(sorted_buffer.data[index:])

    def sample(self, size, replace=False, p=None):
        if not isinstance(p, np.ndarray): p = np.ndarray(p)
        p = p / p.sum()
        return np.random.choice(self.buffer, size, replace, p=p)

    def reset(self) -> 'ArrayBuffer':
        return ArrayBuffer(self.buffer.shape)

    def _copy(self, buffer):
        copy = ArrayBuffer(self.buffer.shape)
        copy.buffer = buffer
        copy.size = min(len(buffer), self.size)
        copy.cnt = self.cnt
        return copy
