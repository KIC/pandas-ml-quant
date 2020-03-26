from typing import Tuple, Callable

import numpy as np
import pandas as pd


class Splitter(object):

    def __init__(self):
        pass

    def train_test_split(self, index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        pass

    @staticmethod
    def infinity_sample_range():
        while True:
            yield

    @property
    def cross_validation(self) -> Tuple[int, Callable[[pd.Index, pd.Index], Tuple[np.ndarray, np.ndarray]]]:
        return None
