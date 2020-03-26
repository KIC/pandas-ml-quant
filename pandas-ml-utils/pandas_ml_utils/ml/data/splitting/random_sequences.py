from typing import Tuple

import pandas as pd
import numpy as np

from pandas_ml_common.utils.random import temp_seed
from pandas_ml_utils.ml.data.splitting.splitter import Splitter


class RandomSequences(Splitter):

    def __init__(self, test_size, session_size=0.7, folds=100, seed=None):
        super().__init__()
        self.test_size = test_size
        self.session_size = session_size
        self.folds = folds
        self.seed = seed

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        # we just split the sequence int past and recent data
        end_idx = int(len(index) * (1 - self.test_size))
        return index[0:end_idx], index[end_idx:]

    @property
    def cross_validation(self):
        # this is the magic part of this splitter, because we randomly start from the taining set only moving forward
        def fold(features, labels):
            # FIXME use session size
            idx = np.random.choice(len(features - 2))
            return features[:idx], features[idx:]

        return fold

