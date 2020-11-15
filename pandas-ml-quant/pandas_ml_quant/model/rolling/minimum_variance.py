from functools import wraps
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.utils import wrap_row_level_as_nested_array
from pandas_ml_utils import Model, Summary, PostProcessedFeaturesAndLabels
from ..rolling_model import RollingModel
from ...technichal_analysis import ta_ewma_covariance


class MinVarianceBaseModel(Model):

    def __init__(self,
                 price_column: str = 'Close',
                 covariance_estimator: Callable = ta_ewma_covariance,
                 long_only: bool = True,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(
            PostProcessedFeaturesAndLabels(
                features=[price_column],
                feature_post_processor=[covariance_estimator, wrap_row_level_as_nested_array],
                targets=[price_column],
                labels=[price_column],
            ),
            summary_provider,
            **kwargs
        )
        self.long_only = long_only
        self._last_loss = np.nan
        self._last_solution = np.nan

    def fit_batch(self, X: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        n = y.shape[1]
        w = cp.Variable(n)
        A = np.ones(n)
        constraints = [np.eye(n) @ w >= 0, A.T @ w == 1] if self.long_only else [A.T @ w == 1]
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(w, X._.values.squeeze())), constraints)
        prob.solve()

        self._last_loss = prob.value
        self._last_solution = w.value

    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        return self._last_loss

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        return pd.DataFrame([self._last_solution], index=features.index, columns=self._labels_columns)


@wraps(MinVarianceBaseModel.__init__)
def MinVarianceModel(*args, **kwargs):
    return RollingModel(MinVarianceBaseModel(*args, **kwargs), window=1, retrain_after=0)