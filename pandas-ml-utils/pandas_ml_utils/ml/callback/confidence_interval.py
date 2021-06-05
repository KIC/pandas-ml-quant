from typing import List, Callable

import pandas as pd

from pandas_ml_common import LazyInit
from ..confidence import CdfConfidenceInterval


class TestConfidenceInterval(object):

    CdfConfidenceInterval = CdfConfidenceInterval

    def __init__(self, ci_provider: CdfConfidenceInterval, mod=None, print=True, early_stopping=False):
        self.ci = ci_provider
        self.call_counter = 0
        self.mod = mod
        self.print = print
        self.early_stopping = early_stopping

    def __call__(self, epoch, y_train: pd.DataFrame, y_test: List, y_hat_train: LazyInit, y_hat_test: List[LazyInit]):
        self.call_counter += 1
        if self.mod is not None and self.call_counter % self.mod != 0:
            return

        train_ci = self.ci.apply(y_hat_train().join(y_train, lsuffix="params"))
        test_ci = [self.ci.apply(y_hat_test[i]().join(y_test[i], lsuffix="params")) for i in range(len(y_test))]

        if self.print:
            print("train:", train_ci, "test:", test_ci)

        if self.early_stopping:
            if max(test_ci) > self.ci.max_tail_events:
                raise StopIteration()
