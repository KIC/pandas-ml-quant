from typing import Tuple, List
import numpy as np


class Buffer(object):

    def sample(self, size) -> Tuple[List, ...]:
        raise NotImplemented()

    def append_args(self, *row) -> 'Buffer':
        return self.append_row(row)

    def append_row(self, row: np.ndarray) -> 'Buffer':
        raise NotImplemented()

    def reset(self) -> 'Buffer':
        raise NotImplemented()

    def __getitem__(self, item):
        # return the column requested
        raise NotImplemented()

    def __len__(self):
        raise NotImplemented()