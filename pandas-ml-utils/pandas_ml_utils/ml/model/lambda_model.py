from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from pandas_ml_common import MlTypes, XYWeight
from pandas_ml_common.utils import call_callable_dynamic_args
from .base_model import ModelProvider
from ..fitting import FittingParameter

_log = logging.getLogger(__name__)


class LambdaModel(ModelProvider):
    """
    The Lambda Model provides a constant output of a given function, hence it can not be trained
    """

    def __init__(self,
                 model: Callable[[pd.DataFrame], np.ndarray],
                 **kwargs):
        super().__init__()
        self.model = model
        self.kwargs = kwargs

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        pass

    def init_fold(self, epoch: int, fold: int):
        pass

    def after_epoch(self, epoch, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        pass

    def finish_learning(self, **kwargs):
        pass

    def encode(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        pass

    def decode(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        pass

    def fit_batch(self, xyw: XYWeight, **kwargs):
        return 0

    def calculate_loss(self, xyw: XYWeight) -> float:
        return 0

    def predict(self, features: List[MlTypes.PatchedDataFrame], samples=1, **kwargs) -> np.ndarray:
        return call_callable_dynamic_args(self.model, features, **self.kwargs)
