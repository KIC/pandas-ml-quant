from typing import Generator, Tuple

import numpy as np


class RollingWindowCV(object):

    def __init__(self, window: int = 200, retrain_after: int = 7):
        super().__init__()
        assert retrain_after >= 0, "forecast need to be > 0"
        self.window = window
        self.retrain_after = retrain_after

    def _split(self, idx, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        full_window_len = self.window + self.retrain_after
        step_size = max(self.retrain_after, 1)

        if len(idx) < full_window_len:
            raise ValueError(f"Not enough Data: {len(idx)} samples < window + forecast: {full_window_len}")

        # FIXME the last training loop needs to be without any test data -> test data empty list
        for i in range(full_window_len, len(idx), step_size):
            train_start, train_stop = i - full_window_len, i - self.retrain_after
            test_start, test_stop = i - step_size, i
            yield np.arange(train_start, train_stop, dtype='int'), np.arange(test_start, test_stop, dtype='int')

    def split(self, idx, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if len(idx) < self.window:
            raise ValueError(f"Not enough Data: {len(idx)} samples < window: {self.window}")

        window = self.window
        step_size = max(self.retrain_after, 1)

        for i in range(window, len(idx) + max(1, step_size - 1), step_size):
            train_start, train_stop = i - window, i
            test_start, test_stop = i, min(i + step_size, len(idx))
            yield np.arange(train_start, train_stop, dtype='int'), np.arange(test_start, test_stop, dtype='int')


