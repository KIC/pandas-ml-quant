from typing import Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from pandas_ml_utils.ml.data.splitting.splitter import Splitter


class NaiveSplitter(Splitter):

    def __init__(self, test_size=0.3, nr_of_splits=None, epochs_per_split=1):
        super().__init__()
        self.test_size = test_size
        self._cross_validation = TimeSeriesSplit(nr_of_splits) if nr_of_splits is not None else None
        self._epochs_per_split = epochs_per_split

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        # we just split the sequence int past and recent data
        end_idx = int(len(index) * (1 - self.test_size))
        return index[0:end_idx], index[end_idx:]

    @property
    def cross_validation(self) -> Tuple[int, Callable[[pd.Index, pd.Index], Tuple[np.ndarray, np.ndarray]]]:
        if self._cross_validation is None:
            return None
        else:
            return (self._epochs_per_split, self._cross_validation.split)


# create an alias for convenience
TimeSeriesSplitter = NaiveSplitter