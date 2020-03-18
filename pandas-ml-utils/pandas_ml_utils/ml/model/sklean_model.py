from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
_log = logging.getLogger(__name__)


class SkModel(Model):

    def __init__(self,
                 skit_model,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.skit_model = skit_model
        self.label_shape = None

    def fit(self,
            x: np.ndarray, y: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray,
            sample_weight_train: np.ndarray, sample_weight_test: np.ndarray) -> float:
        # shape correction if needed
        y = y.reshape((len(x), -1)) if len(y.shape) > 1 and y.shape[1] == 1 else y
        self.label_shape = y.shape

        # remember fitted model
        self.skit_model = self.skit_model.fit(SkModel.reshape_rnn_as_ar(x), y)

        if getattr(self.skit_model, 'loss_', None):
            return self.skit_model.loss_
        else:
            prediction = self.predict(x)
            if isinstance(self.skit_model, LogisticRegression)\
            or type(self.skit_model).__name__.endswith("Classifier")\
            or type(self.skit_model).__name__.endswith("SVC"):
                from sklearn.metrics import log_loss
                try:
                    return log_loss(prediction > 0.5, y).mean()
                except ValueError as e:
                    if "contains only one label" in str(e):
                        return -100
                    else:
                        raise e
            else:
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(prediction, y).mean()

    def predict(self, x) -> np.ndarray:
        if callable(getattr(self.skit_model, 'predict_proba', None)):
            y_hat = self.skit_model.predict_proba(SkModel.reshape_rnn_as_ar(x))
            binary_classifier = len(self.label_shape) == 1 or self.label_shape[1] == 1
            return y_hat[:, 1] if binary_classifier else y_hat.reshape(-1, *self.label_shape[1:])
        else:
            return self.skit_model.predict(SkModel.reshape_rnn_as_ar(x))

    def plot_loss(self):
        loss_curve = getattr(self.skit_model, 'loss_curve_', None)

        if loss_curve is not None:
            import matplotlib.pyplot as plt
            plt.plot(loss_curve)
        else:
            print("no loss curve found")

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
        if len(arr3d.shape) < 3:
            print("Data was not in RNN shape")
            return arr3d
        else:
            return arr3d.reshape(arr3d.shape[0], arr3d.shape[1] * arr3d.shape[2])
