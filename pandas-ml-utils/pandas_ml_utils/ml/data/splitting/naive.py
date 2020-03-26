from typing import Tuple

import pandas as pd

from pandas_ml_utils.ml.data.splitting.splitter import Splitter


class NaiveSplitter(Splitter):

    def __init__(self, test_size=0.3):
        super().__init__()
        self.test_size = test_size

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        # we just split the sequence int past and recent data
        end_idx = int(len(index) * (1 - self.test_size))
        return index[0:end_idx], index[end_idx:]

