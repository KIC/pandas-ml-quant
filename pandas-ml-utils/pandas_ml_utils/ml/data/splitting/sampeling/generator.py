import logging
from typing import Tuple, Callable

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_if_not_none

_log = logging.getLogger(__name__)


class DataGenerator(object):
    # the idea is to pass a yielding data generator to the "fit" method instad of x/y values
    # the cross validation loop goes as default implementation into to Model class.
    # this way each model can implement cross validation on their own and we can use the data generator for the gym

    def __init__(self,
                 cross_validation: Callable,
                 train: Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.DataFrame],
                 test: Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.DataFrame]):
        self.cross_validation = cross_validation
        self.train_x = train[0]
        self.train_y = train[1]
        self.train_w = train[2]
        self.test_x = test[0]
        self.test_y = test[1]
        self.test_w = test[2]

    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_train, y_train, w_train = self.train_x.ml.values, self.train_y.ml.values, call_if_not_none(self.train_w, "values")
        x_test, y_test, w_test = self.test_x.ml.values, self.test_y.ml.values, call_if_not_none(self.test_w, "values")
        cv = self.cross_validation

        # loop through folds and yield data until done then raise StopIteration
        if cv is not None and isinstance(cv, Tuple) and callable(cv[1]):
            for fold_epoch in range(cv[0]):
                # cross validation, make sure we re-shuffle every fold_epoch
                for f, (train_idx, test_idx) in enumerate(cv[1](x_train, y_train)):
                    _log.info(f'fit fold {f}')
                    yield (x_train[train_idx], y_train[train_idx],
                           x_train[test_idx], y_train[test_idx],
                           *((w_train[train_idx], w_train[test_idx]) if w_train is not None else (None, None)))
        else:
            # fit without cross validation
            yield x_train, y_train, x_test, y_test, w_train, w_test

