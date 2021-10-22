from typing import List, Callable, Any

import numpy as np
import pandas as pd

from pandas_ml_common import LazyInit
from pandas_ml_common.utils.numpy_utils import mean
from ..confidence import CdfConfidenceInterval


class TestConfidenceInterval(object):

    CdfConfidenceInterval = CdfConfidenceInterval

    def __init__(self,
                 ci_provider: CdfConfidenceInterval,
                 variance_provider: Callable[[Any], float] = None,
                 mod=None,
                 print=True,
                 early_stopping=False):
        self.ci = ci_provider
        self.variance_provider = variance_provider
        self.call_counter = 0
        self.mod = mod
        self.print = print
        self.early_stopping = early_stopping
        self._history = []

    def __call__(self, epoch, y_train: pd.DataFrame, y_test: List, y_hat_train: LazyInit[np.ndarray], y_hat_test: LazyInit[np.ndarray]):
        self.call_counter += 1
        if self.mod is not None and self.call_counter % self.mod != 0:
            return

        # create the required dataframes for the CdfConfidenceInterval provider which needs 2 columns
        #  the distribution parameters (as list) and the true value
        train_ci_df = pd.DataFrame(y_hat_train(), index=y_train[0].index).agg(lambda x: list(x), axis=1).to_frame().join(y_train[0], lsuffix="params")
        test_ci_dfs = [pd.DataFrame(y_hat_test(), index=yt[0].index).agg(lambda x: list(x), axis=1).to_frame().join(yt[0], lsuffix="params") for yt in y_test]
        train_ci = self.ci.apply(train_ci_df)
        test_ci = mean([self.ci.apply(f) for f in test_ci_dfs])

        if self.variance_provider is not None:
            train_var = train_ci_df.apply(self.variance_provider, axis=1).mean().item()
            test_var = mean([f.apply(self.variance_provider, axis=1).mean().item() for f in test_ci_dfs])
        else:
            train_var = np.nan
            test_var = np.nan

        self._history.append([train_ci, test_ci, train_var, test_var])

        if self.print:
            print(f"train confidence: {train_ci:.6f}, variance: {train_var:.6f}, "
                  f"test confidence: {test_ci:.6f}, variance: {test_var:.6f}"
                  f"\t(epoch: {epoch})")

        if self.early_stopping:
            if test_ci > self.ci.max_tail_events:
                raise StopIteration()

    @property
    def history(self):
        return pd.DataFrame(
            self._history,
            columns=["Train Confidence", "Test Confidence", "Train Variance", "Test Variance"]
        )
