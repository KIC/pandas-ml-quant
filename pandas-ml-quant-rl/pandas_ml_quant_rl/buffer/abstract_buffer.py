from typing import Tuple
import numpy as np


class Buffer(object):

    def rank_and_cut(self, percentile: Tuple[int, float]) -> 'Buffer':
        raise NotImplemented()

    @property
    def is_full(self) -> bool:
        raise NotImplemented()

    @property
    def data(self) -> np.ndarray:
        raise NotImplemented()

    def sample(self, size, replace=False) -> np.ndarray:
        raise NotImplemented()

    def append_args(self, *row) -> 'Buffer':
        return self.append_row(row)

    def append_row(self, row: np.ndarray) -> 'Buffer':
        raise NotImplemented()

    def reset(self) -> 'Buffer':
        raise NotImplemented()