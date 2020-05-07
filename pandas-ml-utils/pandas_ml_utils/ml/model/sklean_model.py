from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from pandas_ml_common import Typing
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)
ConvergenceWarning('ignore')


class SkModel(Model):

    def __init__(self,
                 skit_model,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.skit_model = skit_model
        self.label_shape = None

    def fit_fold(self,
                 fold_nr: int,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        # shape correction if needed
        x = SkModel.reshape_rnn_as_ar(x)
        y = y.reshape((len(x), -1)) if y.ndim > 1 and y.shape[1] == 1 else y
        y = y.reshape(len(x)) if y.ndim == 2 and y.shape[1] == 1 else y
        self.label_shape = y.shape

        # remember fitted model
        self.skit_model = self.skit_model.fit(x, y)

        if getattr(self.skit_model, 'loss_', None):
            loss_curve = getattr(self.skit_model, 'loss_curve_', [])
            return np.array(loss_curve), np.array([])
        else:
            prediction = self.predict_sample(x)
            if isinstance(self.skit_model, LogisticRegression)\
            or type(self.skit_model).__name__.endswith("Classifier")\
            or type(self.skit_model).__name__.endswith("SVC"):
                from sklearn.metrics import log_loss
                try:
                    return np.array(log_loss(prediction > 0.5, y).mean()), np.array([])
                except ValueError as e:
                    if "contains only one label" in str(e):
                        return np.array([-100]), np.array([])
                    else:
                        raise e
            else:
                from sklearn.metrics import mean_squared_error
                return np.array(mean_squared_error(prediction, y).mean()), np.array([])

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if callable(getattr(self.skit_model, 'predict_proba', None)):
            y_hat = self.skit_model.predict_proba(SkModel.reshape_rnn_as_ar(x))
            binary_classifier = len(self.label_shape) == 1 or self.label_shape[1] == 1
            return y_hat[:, 1] if binary_classifier else y_hat.reshape(-1, *self.label_shape[1:])
        else:
            return self.skit_model.predict(SkModel.reshape_rnn_as_ar(x))

    def __str__(self):
        return f'{__name__}({repr(self.skit_model)}, {self.features_and_labels})'

    def __call__(self, *args, **kwargs):
        if not kwargs:
            return deepcopy(self)
        else:
            new_model = SkModel(type(self.skit_model)(**kwargs), self.features_and_labels, self.summary_provider)
            new_model.kwargs = deepcopy(self.kwargs)
            return new_model

    @staticmethod
    def reshape_rnn_as_ar(arr3d):
        if arr3d.ndim < 3:
            print("Data was not in RNN shape")
            return arr3d
        else:
            return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])
