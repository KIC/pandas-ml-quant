
from typing import Tuple, Callable

import numpy as np
import pandas as pd

from pandas_ml_utils.ml.data.splitting.splitter import Splitter


class DummySplitter(Splitter):

    def __init__(self, samples_size=1):
        super().__init__()
        self.sample_size = samples_size

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        return index, index[[]]

    @property
    def cross_validation(self) -> Tuple[int, Callable[[pd.Index, pd.Index], Tuple[np.ndarray, np.ndarray]]]:
        def just_repeat(i1, i2):
            idx = np.arange(len(i1))
            for _ in range(self.sample_size):
                yield idx, idx

        return 1, just_repeat
