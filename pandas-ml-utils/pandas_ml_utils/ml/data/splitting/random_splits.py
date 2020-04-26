import logging
from typing import Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_train_test_split

from pandas_ml_utils.ml.data.splitting.splitter import Splitter

_log = logging.getLogger(__name__)


class RandomSplits(Splitter):

    def __init__(self,
                 test_size: float = 0.4,
                 youngest_size: float = None,
                 cross_validation: Tuple[int, Callable[[pd.Index, pd.Index], Tuple[np.ndarray, np.ndarray]]] = None,
                 test_validate_split_seed=42):
        """
        :param test_size: the fraction [0, 1] of random samples which are used for a test set
        :param youngest_size: the fraction [0, 1] of the test samples which are not random but are the youngest
        :param cross_validation: tuple of number of epochs for each fold provider and a cross validation provider
        :param test_validate_split_seed: seed if train, test splitting needs to be reproduceable. A magic seed 'youngest' is
                                         available, which just uses the youngest data as test data
        """

        super().__init__()
        self.test_size = test_size
        self.youngest_size = test_size * youngest_size if youngest_size is not None else None
        self._cross_validation = cross_validation
        self.seed = test_validate_split_seed

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:

        # convert data frame index to numpy array
        index = index.values

        if self.test_size <= 0:
            train, test = index, index[:0]
        elif self.seed == 'youngest':
            i = int(len(index) - len(index) * self.test_size)
            train, test = index[:i], index[i:]
        else:
            random_sample_test_size = self.test_size if self.youngest_size is None else self.test_size * (1 - self.youngest_size)
            random_sample_train_index_size = int(len(index) - len(index) * (self.test_size - random_sample_test_size))

            if random_sample_train_index_size < len(index):
                _log.warning(f"keeping youngest {len(index) - random_sample_train_index_size} elements in test set")

                # cut the youngest data and use residual to randomize train/test data
                index_train, index_test = \
                    sk_train_test_split(index[:random_sample_train_index_size],
                                     test_size=random_sample_test_size, random_state=self.seed)

                # then concatenate (add back) the youngest data to the random test data
                index_test = np.hstack([index_test, index[random_sample_train_index_size:]])  # index is 1D

                train, test = index_train, index_test
            else:
                train, test = sk_train_test_split(index, test_size=random_sample_test_size, random_state=self.seed)

        return pd.Index(train), pd.Index(test)

    @property
    def cross_validation(self) -> Tuple[int, Callable[[pd.Index, pd.Index], Tuple[np.ndarray, np.ndarray]]]:
        return self._cross_validation

