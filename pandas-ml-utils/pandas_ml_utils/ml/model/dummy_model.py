from __future__ import annotations

import logging
import math
import sys
from copy import deepcopy
from typing import List, Callable, TYPE_CHECKING, Type, Dict, Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)


class DummyModel(Model):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)

    def fit_fold(self,
                 fold_nr: int,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return x

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return x

