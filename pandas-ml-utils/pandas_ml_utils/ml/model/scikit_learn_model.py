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

    def init_fold(self, epoch: int, fold: int):
        if fold > 1:
            self.log_once("merge_folds", _log.warning,
                          "merging of cross folded models is not supported, we just keep training the same model")

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, weight: pd.DataFrame, fold: int, **kwargs):
        # convert data frames to numpy arrays
        _x = _AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(x, split_multi_index_rows=False))
        _y = unpack_nested_arrays(y, split_multi_index_rows=False)
        _w = unpack_nested_arrays(weight, split_multi_index_rows=False)

        _y = _y.reshape((len(_x), -1)) if _y.ndim > 1 and _y.shape[1] == 1 else _y
        _y = _y.reshape(len(_x)) if _y.ndim == 2 and _y.shape[1] == 1 else _y
        if self._label_shape is None: self._label_shape = _y.shape

        par = self._fit_meta_data
        partial_fit = any([size > 1 for size in [par.epochs, par.batch_size, par.fold_epochs] if size is not None])

        if partial_fit:
            # use partial fit whenever possible partial_fit
            if hasattr(self.sk_model, "partial_fit"):
                kw_classes = {"classes": kwargs["classes"]} if "classes" in kwargs else {}
                try:
                    self.sk_model = self.sk_model.partial_fit(_x, _y, **kw_classes)
                except Exception as e:
                    if "classes" in kwargs:
                        raise e
                    else:
                        raise ValueError("You might need to pass 'classes' argument for partial fitting", e)
            else:
                raise ValueError(f"This of model does not support `partial_fit` {type(self.sk_model)} - "
                                 f"and therefore does not support epochs or batches.")
        else:
            self.sk_model = self.sk_model.fit(_x, _y)

    def merge_folds(self, epoch: int):
        pass

    def finish_learning(self):
        # clear oll intermediate fold models, otherwise they get serialized to disk
        pass

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
        skm = self.sk_model
        y_pred = self._predict(skm, x, fold=fold)
        y_true = unpack_nested_arrays(y_true, split_multi_index_rows=False).reshape(y_pred.shape)
        w = weight.values.reshape(-1, ) if weight is not None else None

        if isinstance(self.sk_model, ClassifierMixin):
            # calculate: # sklearn.metrics.log_loss
            return metrics.log_loss(y_true, y_pred, sample_weight=w)
        else:
            # calculate: metrics.mean_squared_error
            return metrics.mean_squared_error(y_true, y_pred, sample_weight=w)

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame=None, latent: pd.DataFrame=None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        return to_pandas(self._predict(self.sk_model, features, samples, **kwargs), features.index, self._labels_columns)

    def _predict(self, skm, features: pd.DataFrame, samples=1, **kwargs) -> np.ndarray:
        x = _AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(features, split_multi_index_rows=False))
        is_probabilistic = callable(getattr(skm, 'predict_proba', None))

        def predictor():
            if is_probabilistic:
                y_hat = skm.predict_proba(x)
                binary_classifier = len(self._label_shape) == 1 or self._label_shape[1] == 1
                return (1 - y_hat[:, 0]) if binary_classifier else y_hat.reshape(-1, *self._label_shape[1:])
            else:
                return skm.predict(x)

        return np.array([predictor() for _ in range(samples)]).swapaxes(0, 1) if samples > 1 else predictor()


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

    def calculate_loss(self, fold, x, y_true, weight) -> float:
        skm = self.sk_model
        y_pred = skm.predict(_AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(x)))
        y_true = unpack_nested_arrays(y_true, split_multi_index_rows=False).reshape(y_pred.shape)
        w = weight.values.reshape(-1, ) if weight is not None else None

        return metrics.mean_squared_error(y_true, y_pred, sample_weight=w)


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

    def _decode(self, latent_features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        skm = self.sk_model
        if not hasattr(skm, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        decoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.decoder_layers, **self.kwargs})
        decoder.coefs_ = skm.coefs_[len(self.encoder_layers):].copy()
        decoder.intercepts_ = skm.intercepts_[len(self.encoder_layers):].copy()
        decoder.n_layers_ = len(decoder.coefs_) + 1
        decoder.n_outputs_ = self.layers[-1]
        decoder.out_activation_ = skm.out_activation_

        decoded = decoder.predict(_AbstractSkModel.reshape_rnn_as_ar(unpack_nested_arrays(latent_features, split_multi_index_rows=False)))
        return to_pandas(decoded, latent_features.index, self._feature_columns)

