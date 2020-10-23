from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, Tuple, List, Union, Generator

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone, ClusterMixin, ClassifierMixin, RegressorMixin
from sklearn import metrics

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args, unpack_nested_arrays, merge_kwargs, to_pandas
from pandas_ml_common.utils.logging_utils import LogOnce
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model, AutoEncoderModel

_log = logging.getLogger(__name__)
ConvergenceWarning('ignore')


class _AbstractSkModel(Model):

    def __init__(self,
                 skit_model,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.sk_model = skit_model
        self.overwrite_folds = True  # maybe later we can handle the folding of models (eventually just taking the best)
        self.log_once = LogOnce().log
        self._label_shape = None
        self._sk_fold_models = []

    def init_fold(self, epoch: int, fold: int):
        i = 0 if not self.overwrite_folds else fold
        if len(self._sk_fold_models) <= i:
            self._sk_fold_models.append(clone(self.sk_model))
        else:
            self._sk_fold_models[i] = clone(self.sk_model)

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, weight: pd.DataFrame, **kwargs):
        # convert data frames to numpy arrays
        _x = _AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(x, split_multi_index_rows=False))
        _y = unpack_nested_arrays(y, split_multi_index_rows=False)
        _w = unpack_nested_arrays(weight, split_multi_index_rows=False)

        _y = _y.reshape((len(_x), -1)) if _y.ndim > 1 and _y.shape[1] == 1 else _y
        _y = _y.reshape(len(_x)) if _y.ndim == 2 and _y.shape[1] == 1 else _y
        if self._label_shape is None: self._label_shape = _y.shape

        # use partial fit whenever possible partial_fit
        if self._fit_meta_data.partial_fit:
            if hasattr(self._sk_fold_models[-1], "partial_fit"):
                kwa = {"classes": kwargs["classes"]} if "classes" in kwargs else {}
                self._sk_fold_models[-1] = self._sk_fold_models[-1].partial_fit(_x, _y, **kwa)
            else:
                raise ValueError(f"This of model does not support `partial_fit` {type(self.sk_model)} - "
                                 f"and therefore does not support epochs or batches.")
        else:
            self._sk_fold_models[-1] = self._sk_fold_models[-1].fit(_x, _y)

    def merge_folds(self, epoch: int):
        self.log_once("merge_folds", _log.warning, "merging of cross folded models is not supported, we just keep training the same model")
        self.sk_model = self._sk_fold_models[-1]

    def finish_learning(self):
        # clear oll intermediate fold models, otherwise they get serialized to disk
        self._sk_fold_models = []

    @staticmethod
    def reshape_rnn_as_ar(arr3d):
        if arr3d.ndim < 3:
            return arr3d
        else:
            return arr3d.reshape(arr3d.shape[0], np.array(arr3d.shape[1:]).prod())



class SkModel(_AbstractSkModel):

    def __init__(self,
                 skit_model,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(skit_model, features_and_labels, summary_provider, **kwargs)

    def calculate_loss(self, fold, x, y_true, weight):
        skm = self.sk_model if fold is None else self._sk_fold_models[0 if not self.overwrite_folds else fold]
        y_pred = self._predict(skm, x, fold=fold)
        y_true = unpack_nested_arrays(y_true, split_multi_index_rows=False).reshape(y_pred.shape)

        if isinstance(self.sk_model, ClassifierMixin):
            # calculate: # sklearn.metrics.log_loss
            return metrics.log_loss(y_true, y_pred, sample_weight=weight)
        else:
            # calculate: metrics.mean_squared_error
            return metrics.mean_squared_error(y_true, y_pred, sample_weight=weight)

    def predict(self, features: pd.DataFrame, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        # FIXME if auto encoder do not call this method ->     # TODO factor this method out
        return to_pandas(self._predict(self.sk_model, features, samples, **kwargs), features.index, self._labels_columns)

    def _predict(self, skm, features: pd.DataFrame, samples=1, **kwargs) -> np.ndarray:
        x = _AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(features, split_multi_index_rows=False))
        is_probabilistic = callable(getattr(skm, 'predict_proba', None))

        def predictor():
            if is_probabilistic:
                y_hat = skm.predict_proba(x)
                binary_classifier = len(self._label_shape) == 1 or self._label_shape[1] == 1
                return y_hat[:, 1] if binary_classifier else y_hat.reshape(-1, *self._label_shape[1:])
            else:
                return skm.predict(x)

        return np.array([predictor() for _ in range(samples)]).swapaxes(0, 1) if samples > 1 else predictor()

    def __call__(self, *args, **kwargs):
        return SkModel(
            deepcopy(self.sk_model),
            deepcopy(self.features_and_labels),
            self.summary_provider,
            **merge_kwargs(self.kwargs, kwargs)
        )


class SkAutoEncoderModel(_AbstractSkModel, AutoEncoderModel):

    def __init__(self,
                 encode_layers: List[int],
                 decode_layers: List[int],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(
            skit_model=call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": [*encode_layers, *decode_layers], **kwargs}),
            features_and_labels=features_and_labels,
            summary_provider=summary_provider,
            **kwargs
        )

        # Implementation analog blog: https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html
        self.encoder_layers = encode_layers
        self.decoder_layers = decode_layers
        self.layers = [*encode_layers, *decode_layers]

    def calculate_loss(self, fold, x, y_true, weight):
        # FIXME ...
        #y_pred = self._predict(skm, x, fold=fold)
        #y_true = unpack_nested_arrays(y_true, split_multi_index_rows=False).reshape(y_pred.shape)

        #return metrics.mean_squared_error(y_true, y_pred, sample_weight=weight)
        return 0

    def _auto_encode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        x = _AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(features, split_multi_index_rows=False))
        return to_pandas(self.sk_model.predict(x), features.index, self._labels_columns)

    def _encode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        skm = self.sk_model
        if not hasattr(skm, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        encoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.encoder_layers[1:], **self.kwargs})
        encoder.coefs_ = skm.coefs_[:len(self.encoder_layers)].copy()
        encoder.intercepts_ = skm.intercepts_[:len(self.encoder_layers)].copy()
        encoder.n_layers_ = len(encoder.coefs_) + 1
        encoder.n_outputs_ = len(self.features_and_labels.latent_names)
        encoder.out_activation_ = skm.activation

        encoded = encoder.predict(_AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(features, split_multi_index_rows=False)))
        return to_pandas(encoded, features.index, self._features_and_labels.latent_names)

    def _decode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        skm = self.sk_model
        if not hasattr(skm, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        decoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.decoder_layers, **self.kwargs})
        decoder.coefs_ = skm.coefs_[len(self.encoder_layers):].copy()
        decoder.intercepts_ = skm.intercepts_[len(self.encoder_layers):].copy()
        decoder.n_layers_ = len(decoder.coefs_) + 1
        decoder.n_outputs_ = self.layers[-1]
        decoder.out_activation_ = skm.out_activation_

        return decoder.predict(_AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(features, split_multi_index_rows=False)))

    def __call__(self, *args, **kwargs):
        copy = SkAutoEncoderModel(
            self.encoder_layers,
            self.decoder_layers,
            deepcopy(self.features_and_labels),
            self.summary_provider,
            **merge_kwargs(self.kwargs, kwargs)
        )

        copy.sk_model = deepcopy(self.sk_model)
        return copy

"""
class SkAutoEncoderModel(NumpyAutoEncoderModel):

    def __init__(self,
                 encode_layers: List[int],
                 decode_layers: List[int],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)

        # Implementation analog blog: https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html
        self.encoder_layers = encode_layers
        self.decoder_layers = decode_layers
        self.layers = [*encode_layers, *decode_layers]
        self.auto_encoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.layers, **kwargs})

    def _fold_epoch(self, train, test, nr_epochs, **kwargs) -> Tuple[float, float]:
        raise NotImplemented

    def _fit_epoch_fold(self, fold, train, test, nr_of_folds, nr_epochs, **kwargs) -> Tuple[float, float]:
        if nr_epochs > 1:
            raise ValueError("partial_fit not implemented")

        self.auto_encoder = self.auto_encoder.fit(train[0], train[1])
        loss_curve = getattr(self.auto_encoder, 'loss_curve_', [])
        return np.array(loss_curve), np.array([np.nan] * len(loss_curve))

    def _auto_encode_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self.auto_encoder.predict(SkModel.reshape_rnn_as_ar(x))

    def _encode_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if not hasattr(self.auto_encoder, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        encoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.encoder_layers[1:], **self.kwargs})
        encoder.coefs_ = self.auto_encoder.coefs_[:len(self.encoder_layers)].copy()
        encoder.intercepts_ = self.auto_encoder.intercepts_[:len(self.encoder_layers)].copy()
        encoder.n_layers_ = len(encoder.coefs_) + 1
        encoder.n_outputs_ = len(self.features_and_labels.latent_names)
        encoder.out_activation_ = self.auto_encoder.activation

        return encoder.predict(SkModel.reshape_rnn_as_ar(x))

    def _decode_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if not hasattr(self.auto_encoder, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        decoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.decoder_layers, **self.kwargs})
        decoder.coefs_ = self.auto_encoder.coefs_[len(self.encoder_layers):].copy()
        decoder.intercepts_ = self.auto_encoder.intercepts_[len(self.encoder_layers):].copy()
        decoder.n_layers_ = len(decoder.coefs_) + 1
        decoder.n_outputs_ = self.layers[-1]
        decoder.out_activation_ = self.auto_encoder.out_activation_

        return decoder.predict(SkModel.reshape_rnn_as_ar(x))
"""
