from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import uuid
from copy import deepcopy
from typing import List, Callable, TYPE_CHECKING, Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import merge_kwargs, suitable_kwargs
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model

_log = logging.getLogger(__name__)


class PytorchModel(Model):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)

    pass
