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


class AutoEncoderModel(Model):

    def __init__(self,
                 trainable_model: Model,
                 encoder_provider: Callable[[Model], Callable[[np.ndarray], np.ndarray]],
                 decoder_provider: Callable[[Model], Callable[[np.ndarray], np.ndarray]],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(trainable_model.features_and_labels, summary_provider, **kwargs)
        self.trainable_model = trainable_model
        self.predictor = None
        self.encoder_provider = encoder_provider
        self.decoder_provider = decoder_provider

    def as_encoder(self) -> Model:
        self.predictor = self.encoder_provider(self.trainable_model)
        return self

    def as_decoder(self) -> Model:
        self.predictor = self.decoder_provider(self.trainable_model)
        return self

    def fit_fold(self,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> float:
        loss = self.trainable_model.fit_fold(x, y,x_val, y_val, sample_weight_train, sample_weight_test, **kwargs)
        self.predictor = self.trainable_model.predict_sample
        return loss

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.predictor(x, **kwargs)

    def __call__(self, *args, **kwargs):
        return AutoEncoderModel(
            self.trainable_model(*args, **kwargs),
            self.encoder_provider,
            self.decoder_provider,
            self.summary_provider,
            **self.kwargs
        )
