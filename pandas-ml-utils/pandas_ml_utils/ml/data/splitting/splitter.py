from typing import Tuple

import pandas as pd


class Splitter(object):

    def __init__(self):
        pass

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        pass

    @property
    def cross_validation(self):
        return None
