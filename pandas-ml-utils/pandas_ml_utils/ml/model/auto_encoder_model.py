import logging
from copy import deepcopy
from typing import List, Callable

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler

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
        self.mode = 'train'

    @property
    def features_and_labels(self):
        return self._features_and_labels

    def as_trainable(self) -> Model:
        copy = self()
        copy.mode = 'train'
        copy._features_and_labels = deepcopy(self.trainable_model.features_and_labels)
        return copy

    def as_encoder(self) -> Model:
        fnl_copy = deepcopy(self.trainable_model.features_and_labels)
        fnl_copy.set_label_columns(self.encoded_column_names, True)

        copy = self()
        copy.mode = 'encode'
        copy._features_and_labels = fnl_copy
        return copy

    def as_decoder(self, decoder_features: List[Typing._Selector] = None) -> Model:
        if "bottleneck" in self.trainable_model.features_and_labels.kwargs and decoder_features is None:
            decoder_features = self.trainable_model.features_and_labels.kwargs["bottleneck"]

        fnl_copy = deepcopy(self.trainable_model.features_and_labels)
        fnl_copy._features = decoder_features

        copy = self()
        copy.mode = 'decode'
        copy._features_and_labels = fnl_copy
        return copy

    def fit(self, sampler: Sampler, **kwargs) -> float:
        loss = self.trainable_model.fit(sampler, **kwargs)
        self._history = self.trainable_model._history
        return loss

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if x.ndim > 2 and x.shape[-1] == 1:
            x = x.reshape(x.shape[:-1])

        if self.mode == 'train':
            return self.trainable_model.predict_sample(x, **kwargs)
        elif self.mode == 'encode':
            return self.encoder_provider(self.trainable_model)(x, **kwargs)
        elif self.mode == 'decode':
            return self.decoder_provider(self.trainable_model)(x, **kwargs)

    def plot_loss(self, **kwargs):
        return self.trainable_model.plot_loss(**kwargs)

    def __call__(self, *args, **kwargs):
        copy = AutoEncoderModel(
            self.trainable_model(*args, **kwargs),
            self.encoded_column_names,
            self.encoder_provider,
            self.decoder_provider,
            self.summary_provider,
            **self.kwargs
        )

        copy.mode = self.mode
        return copy
