from typing import Generator, Tuple

import numpy as np


class RollingWindowCV(object):

    def __init__(self, window: int = 200, retrain_after: int = 7):
        super().__init__()
        assert retrain_after >= 0, "forecast need to be > 0"
        self.window = window
        self.retrain_after = retrain_after
        self._nr_of_splits = 0

    def split(self, idx, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if len(idx) < self.window:
            raise ValueError(f"Not enough Data: {len(idx)} samples < window: {self.window}")

        window = self.window
        step_size = max(self.retrain_after, 1)

        for i in range(window, len(idx), step_size):
            train_start, train_stop = i - window, i
            test_start, test_stop = i, min(i + step_size, len(idx))
            yield np.arange(train_start, train_stop, dtype='int'), np.arange(test_start, test_stop, dtype='int')
            self._nr_of_splits += 1

        if i + step_size <= len(idx):
            i += step_size
            yield np.arange(i - window, len(idx), dtype='int'), np.arange(0, 0, dtype='int')
            self._nr_of_splits += 1
