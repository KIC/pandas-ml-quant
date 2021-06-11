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

    def __call__(self, epoch, y_train: pd.DataFrame, y_test: List, y_hat_train: LazyInit, y_hat_test: List[LazyInit]):
        self.call_counter += 1
        if self.mod is not None and self.call_counter % self.mod != 0:
            return

        train_ci = self.ci.apply(y_hat_train().join(y_train, lsuffix="params"))
        test_ci = mean([self.ci.apply(y_hat_test[i]().join(y_test[i], lsuffix="params")) for i in range(len(y_test))])

        if self.variance_provider is not None:
            train_var = y_hat_train().apply(self.variance_provider).mean().item()
            test_var = mean([y_hat_test[i]().apply(self.variance_provider).mean().item() for i in range(len(y_test))])
        else:
            train_var = np.nan
            test_var = np.nan

        self._history.append([train_ci, test_ci, train_var, test_var])

        if self.print:
            print(f"train tail: {train_ci:.6f}, var: {train_var:.6f},test tail: {test_ci:.6f}, var: {test_var:.6f}\t(epoch: {epoch})")

        if self.early_stopping:
            if test_ci > self.ci.max_tail_events:
                raise StopIteration()

    @property
    def history(self):
        return pd.DataFrame(
            self._history,
            columns=["Train Confidence", "Test Confidence", "Train Variance", "Test Variance"]
        )
