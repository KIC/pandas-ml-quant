from typing import Callable

import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.sampling.cross_validation import PartitionedOnRowMultiIndexCV
from pandas_ml_common.sampling.splitter import duplicate_data
from pandas_ml_quant.model.summary.portfolio_weights_summary import PortfolioWeightsSummary
from pandas_ml_utils import Model, FeaturesAndLabels, Summary


class OneOverNModel(Model):

    def __init__(
            self,
            price_column: str = 'Close',
            summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = PortfolioWeightsSummary,
            **kwargs):
        super().__init__(
            FeaturesAndLabels(features=[price_column], labels=[price_column], targets=[price_column]),
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
        return features.apply(lambda row: np.repeat(1 / len(row), len(row)), result_type='broadcast', axis=1)

    def to_fitter_kwargs(self, **kwargs):
        return {
            "model_provider": self,
            "splitter": duplicate_data(),
            **kwargs
        }