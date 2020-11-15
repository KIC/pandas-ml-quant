from typing import Callable

import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_utils import Model, FeaturesAndLabels, Summary


class OneOverNModel(Model):

    def __init__(self, prices=["Close"], summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary, **kwargs):
        super().__init__(
            FeaturesAndLabels(features=prices, labels=prices),
            summary_provider,
            **kwargs
        )

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        # Nothing to do ...
        pass

    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        return 0

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        # return weights = 1 / n
        return features.apply(lambda row: np.repead(1 / len(row), len(row)), axis=1)
