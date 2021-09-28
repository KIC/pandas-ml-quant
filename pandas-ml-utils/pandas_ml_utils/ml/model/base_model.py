import logging
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable, Tuple, List, Dict, Optional, Any

import numpy as np

from pandas_ml_common import MlTypes, FeaturesLabels, call_callable_dynamic_args
from pandas_ml_common.preprocessing.features_labels import FeaturesWithReconstructionTargets
from pandas_ml_common.sampling.sampler import XYWeight, FoldXYWeight
from pandas_ml_common.utils import merge_kwargs
from pandas_ml_utils.ml.fitting.fitting_parameter import FittingParameter
from pandas_ml_utils.ml.model.fittable import Fittable
from pandas_ml_utils.ml.model.predictable import Model

_log = logging.getLogger(__name__)


class ModelProvider(metaclass=ABCMeta):
    """
    The ModelProvider is an class and need to be implemented for each framework which wants to be supported
    (like tensorflow, pytorch, ...)
    """

    def __call__(self, *args, **kwargs):
        """
        returns a copy pf the model with eventually different configuration (kwargs). This is useful i.e for
        hyper parameter tuning, continuing learning of a previously fitted model or for MultiModels

        :param args:
        :param kwargs: arguments which are eventually provided by hyperopt or by different targets
        :return:
        """
        copy = deepcopy(self)
        copy.kwargs = merge_kwargs(copy.kwargs, kwargs) if hasattr(self, 'kwargs') else kwargs
        return copy

    @abstractmethod
    def fit_batch(self, xyw: XYWeight, **kwargs):
        raise NotImplemented

    @abstractmethod
    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        pass

    @abstractmethod
    def init_fold(self, epoch: int, fold: int):
        pass

    @abstractmethod
    def after_epoch(self, epoch, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        pass

    @abstractmethod
    def calculate_loss(self, xyw: XYWeight) -> float:
        pass

    @abstractmethod
    def finish_learning(self, **kwargs):
        pass

    def train_predict(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        return self.predict(features=features, samples=samples, **kwargs)

    @abstractmethod
    def predict(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def encode(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, features: List[MlTypes.PatchedDataFrame], samples: int = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class FittableModel(Fittable):

    def __init__(self,
                 model_provider: Callable[..., ModelProvider],
                 features_and_labels_definition: FeaturesLabels,
                 cross_validation_aggregator: Callable[[np.ndarray], np.ndarray] = partial(np.mean, axis=0),
                 **kwargs):
        super().__init__(features_and_labels_definition, **kwargs)
        self.model_provider = model_provider
        self.cross_validation_aggregator = cross_validation_aggregator
        self._cross_validation_models: Dict[int, ModelProvider] = defaultdict(
            lambda: call_callable_dynamic_args(self.model_provider, **self.kwargs)
        )

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        for m in self._cross_validation_models.values():
            m.init_fit(fitting_parameter, **kwargs)

    def fit_batch(self, xyw: FoldXYWeight, **kwargs):
        self._cross_validation_models[xyw.fold].fit_batch(xyw)

    def after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        self._cross_validation_models[fold].after_epoch(epoch, fold_epoch, train_data, test_data)

    def finish_learning(self, **kwargs):
        for m in self._cross_validation_models.values():
            m.finish_learning(**kwargs)

    def predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        return self.cross_validation_aggregator(
            np.stack([mp.predict(features.features, samples, **kwargs) for mp in self._cross_validation_models.values()], axis=0)
        )

    def train_predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        return self.cross_validation_aggregator(
            np.stack([mp.train_predict(features.features, samples, **kwargs) for mp in self._cross_validation_models.values()], axis=0)
        )

    def calculate_train_test_loss(self, fold: int, train_data: XYWeight, test_data: List[XYWeight]) -> MlTypes.Loss:
        return (
            self._cross_validation_models[fold].calculate_loss(train_data),
            [self._cross_validation_models[fold].calculate_loss(td) for td in test_data],
        )

    def init_fold(self, epoch: int, fold: int):
        pass

    def after_epoch(self, epoch: int):
        pass


class AutoEncoderModel(Fittable):

    # mode constants
    AUTOENCODE = 'autoencode'
    ENCODE = 'encode'
    DECODE = 'decode'

    def __init__(self,
                 model_provider: Callable[..., ModelProvider],
                 features_and_labels_definition: FeaturesLabels,
                 cross_validation_aggregator: Callable[[np.ndarray], np.ndarray] = partial(np.mean, axis=0),
                 **kwargs):
        super().__init__(
            FeaturesLabels(
                # Note that for the AutoEncoder fitting we need Features == Labels so wie make Features -> Features
                features=features_and_labels_definition.features,
                features_postprocessor=features_and_labels_definition.features_postprocessor,
                labels=features_and_labels_definition.features,
                labels_postprocessor=features_and_labels_definition.features_postprocessor,
                sample_weights=features_and_labels_definition.sample_weights,
                sample_weights_postprocessor=features_and_labels_definition.sample_weights_postprocessor,
                gross_loss=None,
                gross_loss_postprocessor=None,
                reconstruction_targets=None,
                reconstruction_targets_postprocessor=None,
                # ??? label_type=features_and_labels_definition.label_type,
            ),
            **kwargs
        )
        self.encoder_features_and_labels_definition = FeaturesLabels(
                # Note that for the AutoEncoder encoding we make Features -> Labels
                features=features_and_labels_definition.features,
                features_postprocessor=features_and_labels_definition.features_postprocessor,
                ## ??? label_type=features_and_labels_definition.label_type,
            )
        self.decoder_features_and_labels_definition = FeaturesLabels(
                # Note that for the AutoEncoder decoding we make Labels -> Features
                features=features_and_labels_definition.labels,
                features_postprocessor=features_and_labels_definition.labels_postprocessor,
                gross_loss=features_and_labels_definition.gross_loss,
                gross_loss_postprocessor=features_and_labels_definition.gross_loss_postprocessor,
                reconstruction_targets=features_and_labels_definition.reconstruction_targets,
                reconstruction_targets_postprocessor=features_and_labels_definition.reconstruction_targets_postprocessor,
                ## ??? label_type=features_and_labels_definition.label_type,
            )
        self.model_provider = model_provider
        self.cross_validation_aggregator = cross_validation_aggregator
        self._cross_validation_models: Dict[int, ModelProvider] = defaultdict(
            lambda: call_callable_dynamic_args(self.model_provider, **self.kwargs)
        )

        self.mode = AutoEncoderModel.AUTOENCODE

    @property
    def features_and_labels_definition(self) -> FeaturesLabels:
        if self.mode == AutoEncoderModel.ENCODE:
            return self.encoder_features_and_labels_definition
        elif self.mode == AutoEncoderModel.DECODE:
            return self.decoder_features_and_labels_definition
        else:
            return self._features_and_labels_definition

    def label_names(self, df: MlTypes.PatchedDataFrame = None) -> List[Any]:
        if self.mode == AutoEncoderModel.ENCODE:
            label_frames = df.ML.extract(self.features_and_labels_definition).extract_labels().labels
            return [(i, col) if len(label_frames) > 1 else col for i, l in enumerate(label_frames) for col in l]
        else:
            return self._label_names

    def as_encoder(self) -> 'AutoEncoderModel':
        self.mode = AutoEncoderModel.ENCODE
        return self

    def as_decoder(self) -> 'AutoEncoderModel':
        self.mode = AutoEncoderModel.DECODE
        return self

    def as_autoencoder(self) -> 'AutoEncoderModel':
        self.mode = AutoEncoderModel.AUTOENCODE
        return self

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        for m in self._cross_validation_models.values():
            m.init_fit(fitting_parameter, **kwargs)

    def fit_batch(self, xyw: FoldXYWeight, **kwargs):
        self._cross_validation_models[xyw.fold].fit_batch(xyw.x, xyw.y, xyw.weight)

    def after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        self._cross_validation_models[fold].after_epoch(epoch, fold_epoch, train_data, test_data)

    def calculate_train_test_loss(self, fold: int, train_data: XYWeight, test_data: List[XYWeight]) -> MlTypes.Loss:
        return (
            self._cross_validation_models[fold].calculate_loss(train_data),
            [self._cross_validation_models[fold].calculate_loss(td) for td in test_data],
        )

    def finish_learning(self, **kwargs):
        for m in self._cross_validation_models.values():
            m.finish_learning(**kwargs)

    def predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        def encode_decode_autoencode(mp: ModelProvider, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
            if self.mode == AutoEncoderModel.ENCODE:
                return mp.predict(features, samples, **kwargs)
            elif self.mode == AutoEncoderModel.DECODE:
                return mp.encode(features, samples, **kwargs)
            else:
                return mp.decode(features, samples, **kwargs)

        return self.cross_validation_aggregator(
            np.stack(
                [encode_decode_autoencode(mp, features, samples, **kwargs) for mp in self._cross_validation_models.values()],
                axis=0)
        )

    def train_predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        return self.predict(features=features, samples=samples, **kwargs)


class ConcatenatedMultiModel(Model):

    pass


