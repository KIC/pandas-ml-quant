from abc import abstractmethod
from functools import wraps
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common.utils import wrap_row_level_as_nested_array, merge_kwargs
from pandas_ml_utils import Model, Summary, PostProcessedFeaturesAndLabels, FeaturesAndLabels
from ..rolling_model import RollingModel
from ..summary.portfolio_weights_summary import PortfolioWeightsSummary
from ...technichal_analysis import ta_ewma_covariance, ta_mean_returns


class MarkowitzBaseModel(Model):

    def __init__(self, features_and_labels: FeaturesAndLabels, summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary, **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self._last_loss = 0
        self._last_solution = 0

    @abstractmethod
    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        raise NotImplemented

    def solve_quadratic_program(self, covariances: pd.DataFrame, expected_returns: pd.DataFrame = None, risk_aversion: float = 0.5, long_only: bool = True):
        sigma = covariances._.values.squeeze()
        n = sigma.shape[1]
        r = np.ones(n) if expected_returns is None else expected_returns.values.squeeze()
        w = cp.Variable(n)
        A = np.ones(n)

        # weighing factors for very cautious to fully risk averse
        l1 = np.sqrt(1 - risk_aversion)
        l2 = risk_aversion

        cost = cp.quad_form(w, l1 * sigma) - l2 * w.T @ r
        constraints = [np.eye(n) @ w >= 0, A.T @ w == 1] if long_only else [A.T @ w == 1]

        prob = cp.Problem(cp.Minimize(0.5 * cost), constraints)
        prob.solve()

        self._last_loss = prob.value
        self._last_solution = w.value

    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        return self._last_loss

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        return pd.DataFrame([self._last_solution], index=features.index, columns=self._labels_columns)


class MinVarianceBaseModel(MarkowitzBaseModel):

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
        self.solve_quadratic_program(X, None, 0.0, self.long_only)


class MarkowitzBaseModel(MarkowitzBaseModel):

    def __init__(self,
                 price_column: str = 'Close',
                 risk_aversion: float = 0.5,
                 covariance_estimator: Callable = ta_ewma_covariance,
                 returns_estimator: Callable = ta_mean_returns,
                 long_only: bool = True,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(
            FeaturesAndLabels(
                features=(
                    [lambda df: wrap_row_level_as_nested_array(covariance_estimator(df[price_column]))],
                    returns_estimator
                ),
                targets=[price_column],
                labels=[price_column],
            ),
            summary_provider,
            **kwargs
        )
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self._last_loss = np.nan
        self._last_solution = np.nan

    def fit_batch(self, X: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        if isinstance(X, MultiFrameDecorator):
            sigma, mu = X.frames(False)
            self.solve_quadratic_program(sigma, mu, self.risk_aversion, self.long_only)
        else:
            self.solve_quadratic_program(X, None, 0.0, self.long_only)


@wraps(MinVarianceBaseModel.__init__)
def MinVarianceModel(*args, **kwargs):
    kwargs = merge_kwargs(kwargs, summary_provider=PortfolioWeightsSummary)
    return RollingModel(MinVarianceBaseModel(*args, **kwargs), window=1, retrain_after=0)


@wraps(MarkowitzBaseModel.__init__)
def MarkowitzModel(*args, **kwargs):
    kwargs = merge_kwargs(kwargs, summary_provider=PortfolioWeightsSummary)
    return RollingModel(MarkowitzBaseModel(*args, **kwargs), window=1, retrain_after=0)

