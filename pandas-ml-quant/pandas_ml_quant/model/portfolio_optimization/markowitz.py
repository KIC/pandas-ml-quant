from typing import List, Callable, Optional, Any

import pandas as pd
import numpy as np

from pandas_ml_common import XYWeight, MlTypes
from pandas_ml_common.preprocessing.features_labels import FeaturesWithReconstructionTargets
from pandas_ml_common.sampling.sampler import FoldXYWeight
from pandas_ml_utils import Model, FeaturesLabels, Fittable, FittingParameter
from pandas_ml_utils.ml.forecast import Forecast


class MarkowitzModel(Model):

    def __init__(self,
                 price_columns: List[Any],
                 expected_returns: Callable[[MlTypes.PatchedDataFrame], MlTypes.PatchedDataFrame],
                 covariance_estimator: Callable[[MlTypes.PatchedDataFrame], MlTypes.PatchedDataFrame],
                 riskaversion: float = 0,
                 add_riskless_cash_as: Optional[Any] = None,
                 max_short_weight: float = 0,
                 max_long_weight: float = 1,
                 l1=0,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        super().__init__(
            FeaturesLabels(
                features=[
                    [lambda df: df.join(pd.Series(1.0, index=df.index, name=add_riskless_cash_as)) if add_riskless_cash_as is not None else df],
                    [lambda df: df.join(pd.Series(1.0, index=df.index, name=add_riskless_cash_as)) if add_riskless_cash_as is not None else df]
                ],
                features_postprocessor=[expected_returns, covariance_estimator],
                reconstruction_targets=price_columns
            ),
            forecast_provider,
            **kwargs
        )
        self.add_riskless_cash = add_riskless_cash_as is not None
        self.riskaversion = riskaversion
        self.max_short_weight = max_short_weight
        self.max_long_weight = max_long_weight
        self.l1 = l1
        self._label_names = price_columns + ([add_riskless_cash_as] if self.add_riskless_cash else [])

    def predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        import cvxpy as cvx

        self.min_required_samples = features.min_required_samples
        expected_returns, covariances = features.features
        n = len(self.label_names)

        # optimization function
        def optimize(mu, sigma):
            # define optimization target
            w = cvx.Variable(n)

            # define optimization problem
            objective = cvx.Minimize(cvx.quad_form(w, sigma) + (self.l1 * cvx.norm(w, 1)) - (self.riskaversion * w.T @ mu))
            constraints = [w >= self.max_short_weight, w <= self.max_long_weight, cvx.sum(w) == 1]
            prob = cvx.Problem(objective, constraints)

            # solve problem and return result
            prob.solve(verbose=False)
            return w.value

        def construct_mu_sigma(i):
            mu, sigma = expected_returns.loc[i].values, covariances.loc[i].ML.values.squeeze()
            return mu, sigma

        # we need to iterate row for row and if there is one row containing nans we return nans
        portfolio_weights = np.array([optimize(*construct_mu_sigma(i))for i in expected_returns.index])
        return portfolio_weights
