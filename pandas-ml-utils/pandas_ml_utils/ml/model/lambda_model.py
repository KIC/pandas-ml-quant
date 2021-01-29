from __future__ import annotations

import logging
from typing import Callable, Union

import numpy as np
import pandas as pd

from pandas_ml_common import Typing
from pandas_ml_common.utils import to_pandas, call_callable_dynamic_args
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)


class LambdaModel(Model):

    def __init__(self,
                 model: Callable[[pd.DataFrame], Union[np.ndarray, pd.DataFrame]],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.model = model

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        pass

    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        return 0

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        pred = call_callable_dynamic_args(self.model, features, targets=targets, **self.kwargs)

        if isinstance(pred, pd.DataFrame):
            pred.columns = self._labels_columns
            return pred
        else:
            return to_pandas(pred, features.index, self._labels_columns)
