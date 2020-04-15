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
                 encoded_column_names: List[str],
                 encoder_provider: Callable[[Model], Callable[[np.ndarray], np.ndarray]],
                 decoder_provider: Callable[[Model], Callable[[np.ndarray], np.ndarray]],
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(trainable_model.features_and_labels, summary_provider, **kwargs)
        self._features_and_labels = trainable_model.features_and_labels
        self.trainable_model = trainable_model
        self.encoded_column_names = encoded_column_names
        self.encoder_provider = encoder_provider
        self.decoder_provider = decoder_provider
        self._predictor = None

    @property
    def features_and_labels(self):
        return self._features_and_labels

    def as_encoder(self) -> Model:
        fnl_copy = deepcopy(self.trainable_model.features_and_labels)
        fnl_copy.set_label_columns(self.encoded_column_names, True)

        return self._copy_keep_original_model(
            self.encoder_provider(self.trainable_model),
            fnl_copy
        )

    def as_decoder(self, decoder_features: List[Typing._Selector]) -> Model:
        fnl_copy = deepcopy(self.trainable_model.features_and_labels)
        fnl_copy._features = decoder_features

        return self._copy_keep_original_model(
            self.decoder_provider(self.trainable_model),
            fnl_copy
        )

    def fit_fold(self,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> float:
        loss = self.trainable_model.fit_fold(x, y,x_val, y_val, sample_weight_train, sample_weight_test, **kwargs)
        self._predictor = self.trainable_model.predict_sample
        return loss

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if len(x.shape) > 2 and x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])

        if self._predictor is None:
            return self._copy_keep_original_model(self.trainable_model.predict_sample,
                                                  self.trainable_model.features_and_labels)\
                       ._predictor(x, **kwargs)
        else:
            return self._predictor(x, **kwargs)

    def _copy_keep_original_model(self, new_predictior, new_features_and_labels):
        copy = AutoEncoderModel(
            self.trainable_model,
            self.encoded_column_names,
            self.encoder_provider,
            self.decoder_provider,
            self.summary_provider,
            **self.kwargs
        )

        copy._predictor = new_predictior
        copy._features_and_labels = new_features_and_labels

        return copy

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_features_and_labels']
        del state['_predictor']
        return state

    def __call__(self, *args, **kwargs):
        return AutoEncoderModel(
            self.trainable_model(*args, **kwargs),
            self.encoded_column_names,
            self.encoder_provider,
            self.decoder_provider,
            self.summary_provider,
            **self.kwargs
        )
