from typing import Tuple

import numpy as np

from .abstract_buffer import Buffer


class ListBuffer(Buffer):

    def __init__(self, columns: int = 1):
        self.columns = [[] for _ in range(columns)]

    def add(self, row):
        return self.append(*row)

    def append(self, *row):
        assert len(row) == len(self.columns)
        for i in range(len(row)):
            self.columns[i].append(row[i])

    def sample(self, size, replace=False, p=None) -> np.ndarray:
        return np.random.choice(self.data, size, replace, p)

    def rank_and_cut(self, percentile: Tuple[int, float]) -> 'ListBuffer':
        column, percentile = percentile
        ranks = sorted(range(len(self.columns[column])), key=self.columns[column].__getitem__)
        index = int(percentile * (len(ranks) / 100) * 100)
        copy = ListBuffer(len(self.columns))

        for i in range(len(self.columns)):
            copy.columns[i] = [self.columns[i][j] for j in ranks[index:]]

        return copy

    def reset(self) -> 'ListBuffer':
        return ListBuffer(len(self.columns))

    @property
    def data(self) -> np.ndarray:
        return np.array(self.columns).T

    @property
    def is_full(self) -> bool:
        return True

    def valid_buffer(self) -> 'ListBuffer':
        return self

    def __len__(self):
        return len(self.columns[0])
