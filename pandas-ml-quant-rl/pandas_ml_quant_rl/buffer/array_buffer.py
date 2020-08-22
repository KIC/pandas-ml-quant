from typing import Tuple

import numpy as np

from .abstract_buffer import Buffer


class ArrayBuffer(Buffer):

    def __init__(self, shape: Tuple[int], dtype='float'):
        self.buffer = np.empty(shape, dtype=dtype)
        self.size = shape[0]
        self.cnt = 0

    def append_row(self, row: np.ndarray) -> 'ArrayBuffer':
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

    def rank_and_cut(self, percentile: Tuple[int, float]) -> 'ArrayBuffer':
        column, percentile = percentile
        sorted_buffer = self._sort_by_column(column)
        index = int(percentile * (len(sorted_buffer) / 100) * 100)
        return self._copy(sorted_buffer.data[index:])

    def _sort_by_column(self, col: int = 0) -> 'ArrayBuffer':
        data = self.buffer if self.is_full else self.buffer[-self.cnt:]
        buffer = self._copy(self.buffer[np.argsort(data[:, col])])
        return buffer

    def sample(self, size, replace=False) -> np.ndarray:
        data = self.buffer if self.is_full else self.buffer[-self.cnt:]
        return data[np.random.choice(np.arange(len(data)), size, replace)]

    def reset(self) -> 'ArrayBuffer':
        return ArrayBuffer(self.buffer.shape)

    def _copy(self, buffer):
        copy = ArrayBuffer(buffer.shape)
        copy.buffer = buffer
        copy.cnt = self.cnt
        return copy

    def __len__(self):
        return min(self.cnt, self.size)