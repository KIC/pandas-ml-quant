from typing import Tuple, Iterable, Union, List

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

    def sample(self, size) -> Tuple[List, ...]:
        indices = np.random.choice(len(self), size)
        return tuple([[c[i] for i in indices] for c in self.columns])

    def reset(self) -> 'ListBuffer':
        return ListBuffer(self.column_names.keys())

    def __len__(self):
        return len(self.columns[0])
