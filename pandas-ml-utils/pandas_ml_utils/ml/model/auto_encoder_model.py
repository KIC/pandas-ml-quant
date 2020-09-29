import logging
from abc import abstractmethod
from typing import Callable, Tuple

import numpy as np

from pandas_ml_common import Sampler, NumpySampler
from pandas_ml_common import Typing
from pandas_ml_common.utils import to_pandas
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model, SamplerFrameConstants, _NumpyModelFit

_log = logging.getLogger(__name__)


class AutoEncoderModel(Model):

    # mode constants
    autoencode = 'autoencode'
    encode = 'encode'
    decode = 'decode'

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.mode = AutoEncoderModel.autoencode

    def _predict(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        if self.mode == AutoEncoderModel.autoencode:
            return self._auto_encode(sampler, **kwargs)
        elif self.mode == AutoEncoderModel.encode:
            return self._encode(sampler, **kwargs)
        elif self.mode == AutoEncoderModel.decode:
            return self._decode(sampler, **kwargs)
        else:
            raise ValueError("Illegal mode")

    def as_auto_encoder(self) -> 'AutoEncoderModel':
        copy = self()
        copy.mode = AutoEncoderModel.autoencode
        return copy

    def as_encoder(self) -> 'AutoEncoderModel':
        copy = self()
        copy.mode = AutoEncoderModel.encode
        return copy

    def as_decoder(self) -> 'AutoEncoderModel':
        copy = self()
        copy.mode = AutoEncoderModel.decode
        return copy

    @abstractmethod
    def _auto_encode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def _encode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def _decode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented


class NumpyAutoEncoderModel(AutoEncoderModel, _NumpyModelFit):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)

    def _fit(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        return super()._fit_with_numpy(sampler)

    def _auto_encode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        ns = NumpySampler(sampler)
        prediction = np.array([self._auto_encode_epoch(t[SamplerFrameConstants.FEATURES], **kwargs) for (_, t, _) in ns.sample_full_epochs()]).swapaxes(0, 1)
        return to_pandas(prediction, sampler.frames[SamplerFrameConstants.FEATURES].index, self.labels_columns)

    def _encode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        ns = NumpySampler(sampler)
        prediction = np.array([self._encode_epoch(t[SamplerFrameConstants.FEATURES], **kwargs) for (_, t, _) in ns.sample_full_epochs()]).swapaxes(0, 1)
        return to_pandas(prediction, sampler.frames[SamplerFrameConstants.FEATURES].index, self._features_and_labels.latent_names)

    def _decode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        ns = NumpySampler(sampler)
        prediction = np.array([self._decode_epoch(t[SamplerFrameConstants.LATENT], **kwargs) for (_, t, _) in ns.sample_full_epochs()]).swapaxes(0, 1)
        return to_pandas(prediction, sampler.frames[SamplerFrameConstants.FEATURES].index, self.labels_columns)

    def _predict_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._auto_encode_epoch(x, **kwargs)

    @abstractmethod
    def _fold_epoch(self, train, test, nr_epochs, **kwargs) -> Tuple[float, float]:
        raise NotImplemented

    @abstractmethod
    def _fit_epoch_fold(self, fold, train, test, nr_of_folds, nr_epochs, **kwargs) -> Tuple[float, float]:
        raise NotImplemented

    @abstractmethod
    def _auto_encode_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def _encode_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplemented

    @abstractmethod
    def _decode_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplemented

