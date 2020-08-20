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

    def valid_buffer(self) -> 'Buffer':
        raise NotImplemented()

    def add(self, row: np.ndarray) -> 'Buffer':
        raise NotImplemented()

    def reset(self) -> 'Buffer':
        raise NotImplemented()