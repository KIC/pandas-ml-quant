from typing import Tuple, Iterable, Union

import numpy as np

from .abstract_buffer import Buffer


class ListBuffer(Buffer):

    def __init__(self, columns: Union[int, Iterable[str]] = 1, max_size:int = None):
        self.column_names = {col: i for i, col in enumerate(list(range(columns)) if isinstance(columns, int) else columns)}
        self.columns = [[] for _ in range(len(self.column_names))]
        self.max_size = None
        self.push_count = 0

    def append_row(self, row: np.ndarray) -> 'ListBuffer':
        assert len(row) == len(self.columns)
        for i, col in enumerate(row):
            if self.max_size is not None and len(self) > self.max_size:
                self.columns[self.push_count % self.max_size].append(col)
            else:
                self.columns[i].append(col)

        self.push_count += 1
        return self

    def __getitem__(self, item):
        return self.columns[item if isinstance(item, int) else self.column_names[item]]

    # obsolete
    def sample(self, size, replace=False, p=None) -> np.ndarray:
        return np.random.choice(self.data, size, replace, p)

    # obsolete
    def rank_and_cut(self, percentile: Tuple[int, float]) -> 'ListBuffer':
        column, percentile = percentile
        ranks = sorted(range(len(self.columns[column])), key=self.columns[column].__getitem__)
        index = int(percentile * (len(ranks) / 100) * 100)
        copy = ListBuffer(len(self.columns))

        for i in range(len(self.columns)):
            copy.columns[i] = [self.columns[i][j] for j in ranks[index:]]

        return copy

    def reset(self) -> 'ListBuffer':
        return ListBuffer(self.column_names.keys())

    # obsolete
    @property
    def data(self) -> np.ndarray:
        return np.array(self.columns).T

    # obsolete
    @property
    def is_full(self) -> bool:
        return True

    # obsolete
    def valid_buffer(self) -> 'ListBuffer':
        return self

    def __len__(self):
        return len(self.columns[0])
