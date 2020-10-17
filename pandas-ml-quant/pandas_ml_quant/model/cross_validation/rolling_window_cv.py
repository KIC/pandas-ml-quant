from typing import Generator, Tuple, List

import pandas as pd


class RollingWindowCV(object):
    # act like a scikit base cross validator:
    #  hasattr(cross_validation, 'get_n_splits') and hasattr(cross_validation, 'split'):

    def __init__(self, window: int = 200, forecast: int = 7):
        super().__init__()
        assert forecast > 0, "forecast need to be > 0"
        self.window = window
        self.forecast = forecast

    def get_n_splits(self, idx=None, y=None, groups=None):
        nr_windows = (len(idx) - self.forecast) - self.window + 1
        return nr_windows

    def split(self, idx, y=None, groups=None) -> Generator[Tuple[List[int], List[int]], None, None]:
        for i in range(self.window - 1, len(idx) - self.forecast):
            start, stop = i - (self.window - 1), i + 1
            train_idx = range(start, stop)
            test_idx = range(start + self.forecast, stop + self.forecast)
            yield list(train_idx), list(test_idx)


