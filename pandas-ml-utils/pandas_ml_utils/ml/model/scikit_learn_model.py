from __future__ import annotations

import logging
from typing import List, Union, Optional

import numpy as np
from sklearn import metrics
from sklearn.base import ClusterMixin, ClassifierMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from pandas_ml_common import XYWeight, MlTypes
from pandas_ml_common.utils import call_callable_dynamic_args, unpack_nested_arrays, to_pandas
from pandas_ml_common.utils.logging_utils import LogOnce
from .base_model import ModelProvider
from ..fitting import FittingParameter

_log = logging.getLogger(__name__)
ConvergenceWarning('ignore')


class SkModelProvider(ModelProvider):

    def __init__(self, scikit_model: Union[RegressorMixin, ClassifierMixin, ClusterMixin]):
        super().__init__()
        self.sk_model = scikit_model
        self.log_once = LogOnce().log
        self._partial_fit = False
        self._label_shape = None

    def fit_batch(self,
                  x: List[MlTypes.PatchedDataFrame],
                  y: List[MlTypes.PatchedDataFrame],
                  weight: Optional[List[MlTypes.PatchedDataFrame]],
                  **kwargs) -> MlTypes.Loss:

        # convert data frames to numpy arrays, since scikit learn models do not support graph representations
        # we simply concatenate eventual multiple features or labels. As a consequence this requires all feature (or
        # label or weight) frames to share the same dimensions
        _x = np.concatenate([reshape_rnn_as_ar(frame.ML.values) for frame in x], axis=-1)
        _y = np.concatenate([reshape_rnn_as_ar(frame.ML.values) for frame in y], axis=-1)
        _w = np.concatenate([reshape_rnn_as_ar(frame.ML.values) for frame in weight], axis=-1) if weight is not None else None

        _y = _y.reshape((len(_x), -1)) if _y.ndim > 1 and _y.shape[1] == 1 else _y
        _y = _y.reshape(len(_x)) if _y.ndim == 2 and _y.shape[1] == 1 else _y
        if self._label_shape is None: self._label_shape = _y.shape

        if self._partial_fit:
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

        return self.calculate_loss(_x, _y, _w)

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        sizes = [fitting_parameter.epochs, fitting_parameter.batch_size, fitting_parameter.fold_epochs]
        self._partial_fit = any([size > 1 for size in sizes if size is not None])

    def calculate_loss(self, x: np.ndarray, y_true: np.ndarray, weight: Optional[np.ndarray]) -> float:
        y_pred = self._predict(x, 1)
        y_true = unpack_nested_arrays(y_true, split_multi_index_rows=False).reshape(y_pred.shape)
        w = weight.reshape(-1, ) if weight is not None else None

        if isinstance(self.sk_model, ClassifierMixin):
            # calculate: # sklearn.metrics.log_loss
            return metrics.log_loss(y_true, y_pred, sample_weight=w)
        else:
            # calculate: metrics.mean_squared_error
            return metrics.mean_squared_error(y_true, y_pred, sample_weight=w)

    def init_fold(self, epoch: int, fold: int):
        pass

    def after_epoch(self, epoch, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        pass

    def finish_learning(self, **kwargs):
        pass

    def predict(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        _x = np.concatenate([reshape_rnn_as_ar(frame.ML.values) for frame in features], axis=-1)
        return self._predict(_x, samples, **kwargs)

    def _predict(self, x: np.ndarray, samples: int, **kwargs) -> np.ndarray:
        is_probabilistic = callable(getattr(self.sk_model, 'predict_proba', None))

        def predictor():
            if is_probabilistic:
                y_hat = self.sk_model.predict_proba(x)
                binary_classifier = len(self._label_shape) == 1 or self._label_shape[1] == 1
                return (1 - y_hat[:, 0]) if binary_classifier else y_hat.reshape(-1, *self._label_shape[1:])
            else:
                return self.sk_model.predict(x)

        return np.array([predictor() for _ in range(samples)]).swapaxes(0, 1) if samples > 1 else predictor()

    def encode(self, features: MlTypes.PatchedDataFrame, samples: int = 1, **kwargs) -> np.ndarray:
        pass

    def decode(self, features: MlTypes.PatchedDataFrame, samples: int = 1, **kwargs) -> np.ndarray:
        pass


class SkAutoEncoderProvider(SkModelProvider):

    def __init__(self, encode_layers: List[int], decode_layers: List[int], **kwargs):
        super().__init__(
            call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": [*encode_layers, *decode_layers], **kwargs}),
        )

        # Implementation analog blog: https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html
        self.encoder_layers = encode_layers
        self.decoder_layers = decode_layers
        self.layers = [*encode_layers, *decode_layers]

    def calculate_loss(self, x: np.ndarray, y_true: np.ndarray, weight: Optional[np.ndarray]) -> float:
        y_pred = self._predict(x, 1)
        y_true = y_true.reshape(y_pred.shape)
        w = weight.reshape(-1, ) if weight is not None else None
        return metrics.mean_squared_error(y_true, y_pred, sample_weight=w)

    def encode(self, features: MlTypes.PatchedDataFrame, samples: int = 1, **kwargs) -> np.ndarray:
        skm = self.sk_model
        if not hasattr(skm, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        encoder = call_callable_dynamic_args(MLPRegressor,
                                             **{"hidden_layer_sizes": self.encoder_layers[1:], **self.kwargs})
        encoder.coefs_ = skm.coefs_[:len(self.encoder_layers)].copy()
        encoder.intercepts_ = skm.intercepts_[:len(self.encoder_layers)].copy()
        encoder.n_layers_ = len(encoder.coefs_) + 1
        encoder.n_outputs_ = len(self.features_and_labels.latent_names)
        encoder.out_activation_ = skm.activation

        return reshape_rnn_as_ar(features.ML.values)

    def decode(self, features: MlTypes.PatchedDataFrame, samples: int = 1, **kwargs) -> np.ndarray:
        skm = self.sk_model
        if not hasattr(skm, 'coefs_'):
            raise ValueError("Model needs to be 'fit' first!")

        decoder = call_callable_dynamic_args(MLPRegressor, **{"hidden_layer_sizes": self.decoder_layers, **self.kwargs})
        decoder.coefs_ = skm.coefs_[len(self.encoder_layers):].copy()
        decoder.intercepts_ = skm.intercepts_[len(self.encoder_layers):].copy()
        decoder.n_layers_ = len(decoder.coefs_) + 1
        decoder.n_outputs_ = self.layers[-1]
        decoder.out_activation_ = skm.out_activation_

        return decoder.predict(reshape_rnn_as_ar(features.ML.values))


def reshape_rnn_as_ar(arr3d):
    if arr3d.ndim < 3:
        return arr3d
    else:
        return arr3d.reshape(arr3d.shape[0], np.array(arr3d.shape[1:]).prod())