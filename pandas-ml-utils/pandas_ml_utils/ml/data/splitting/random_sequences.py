from typing import Tuple, Callable

import pandas as pd
import numpy as np

from pandas_ml_common.utils.random import temp_seed
from pandas_ml_utils.ml.data.splitting.splitter import Splitter


class RandomSequences(Splitter):

    def __init__(self, test_size=0.4, session_size=0.7, max_folds=100, seed=None):
        super().__init__()
        self.test_size = test_size
        self.session_size = session_size
        self.max_folds = max_folds
        self.seed = seed
        self.min_fold_validation_size=2
        self.min_training_samples=2

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        # we just split the sequence int past and recent data
        end_idx = int(len(index) * (1 - self.test_size))
        return index[0:end_idx], index[end_idx:]

    @property
    def cross_validation(self) -> Tuple[int, Callable[[pd.Index, pd.Index], Tuple[np.ndarray, np.ndarray]]]:
        # this is the magic part of this splitter, because we randomly start from the taining set only moving forward
        # if max_folds is None then we keep indefinitely sampling data

        def sampler(features_index, labels_index) -> Tuple[np.ndarray, np.ndarray]:
            # convert pandas index to array index
            index = np.arange(len(features_index))

            # calculate the latest possible index such that we can sample a whole session
            max_idx = int(len(index) * (1 - self.session_size)) - self.min_fold_validation_size - self.min_training_samples
            assert max_idx > 0, f"not enough data! {features_index.shape}"

            # sample the data (make sure we have at least one sample in the training data)
            for i in (range(self.max_folds) if self.max_folds is not None else Splitter.infinity_sample_range()):
                if self.seed is not None:
                    with temp_seed(self.seed):
                        idx = np.random.choice(max_idx)
                else:
                    idx = np.random.choice(max_idx)

                # finally yield the resulting indices
                yield index[:idx+self.min_training_samples], index[idx+self.min_training_samples:]

        return 1, sampler

