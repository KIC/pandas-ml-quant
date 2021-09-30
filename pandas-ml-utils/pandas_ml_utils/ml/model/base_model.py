import logging
from abc import abstractmethod, ABCMeta
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable, Tuple, List, Dict, Optional, Any, Set, Union

import numpy as np

from pandas_ml_common import MlTypes, FeaturesLabels, call_callable_dynamic_args
from pandas_ml_common.preprocessing.features_labels import FeaturesWithReconstructionTargets, FeaturesWithLabels
from pandas_ml_common.sampling.sampler import XYWeight, FoldXYWeight
from pandas_ml_common.utils import merge_kwargs
from pandas_ml_utils.ml.fitting.fitting_parameter import FittingParameter
from pandas_ml_utils.ml.model.fittable import Fittable
from pandas_ml_utils.ml.model.predictable import Model
from ..forecast import Forecast
from ..summary import Summary

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
                 summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        super().__init__(features_and_labels_definition, summary_provider, forecast_provider, **kwargs)
        self.model_provider = model_provider
        self.cross_validation_aggregator = cross_validation_aggregator
        self._cross_validation_models: Dict[int, ModelProvider] = defaultdict(
            lambda: call_callable_dynamic_args(self.model_provider, **self.kwargs)
        )

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        for m in self._cross_validation_models.values():
            m.init_fit(fitting_parameter, **kwargs)

    def fit_batch(self, xyw: FoldXYWeight, **kwargs):
        self._cross_validation_models[xyw.fold].fit_batch(xyw, **kwargs)

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
                 summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
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
                label_type=features_and_labels_definition.label_type[0] if features_and_labels_definition.label_type is not None else None
            ),
            summary_provider,
            forecast_provider,
            **kwargs
        )

        # the primary goal of an AutoEncoder is to encode features, this is how the features and labels should be
        # defined in the first place
        self.encoder_features_and_labels_definition = FeaturesLabels(
            features=features_and_labels_definition.features,
            features_postprocessor=features_and_labels_definition.features_postprocessor,
            labels=features_and_labels_definition.labels,
            labels_postprocessor=features_and_labels_definition.labels_postprocessor,
            sample_weights=features_and_labels_definition.sample_weights,
            sample_weights_postprocessor=features_and_labels_definition.sample_weights_postprocessor,
            gross_loss=features_and_labels_definition.gross_loss,
            gross_loss_postprocessor=features_and_labels_definition.gross_loss_postprocessor,
            reconstruction_targets=features_and_labels_definition.reconstruction_targets,
            reconstruction_targets_postprocessor=features_and_labels_definition.reconstruction_targets_postprocessor,
            label_type=features_and_labels_definition.label_type[1] if features_and_labels_definition.label_type is not None else None
        )

        # decoding is the secondary target and needs to reverse the features and labels definition
        self.decoder_features_and_labels_definition = FeaturesLabels(
                # Note that for the AutoEncoder decoding we make Labels -> Features
                features=features_and_labels_definition.labels,
                features_postprocessor=features_and_labels_definition.labels_postprocessor,
                gross_loss=features_and_labels_definition.gross_loss,
                gross_loss_postprocessor=features_and_labels_definition.gross_loss_postprocessor,
                reconstruction_targets=features_and_labels_definition.reconstruction_targets,
                reconstruction_targets_postprocessor=features_and_labels_definition.reconstruction_targets_postprocessor,
                label_type=features_and_labels_definition.label_type[0] if features_and_labels_definition.label_type is not None else None
            )

        # define the mode of the AutoEncoder initially as AutoEncoding because initially we need to train the AE first
        self.mode = AutoEncoderModel.AUTOENCODE

        # the the rest of the fields
        self.model_provider = model_provider
        self.cross_validation_aggregator = cross_validation_aggregator
        self._cross_validation_models: Dict[int, ModelProvider] = defaultdict(
            lambda: call_callable_dynamic_args(self.model_provider, **self.kwargs)
        )

    @property
    def features_and_labels_definition(self) -> FeaturesLabels:
        if self.mode == AutoEncoderModel.ENCODE:
            return self.encoder_features_and_labels_definition
        elif self.mode == AutoEncoderModel.DECODE:
            return self.decoder_features_and_labels_definition
        else:
            return self._features_and_labels_definition

    @property
    def label_names(self) -> List[Any]:
        if self.mode == AutoEncoderModel.ENCODE:
            return self.features_and_labels_definition.labels
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
        self._cross_validation_models[xyw.fold].fit_batch(xyw)

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
            if self.mode == AutoEncoderModel.AUTOENCODE:
                return mp.predict(features.features, samples, **kwargs)
            elif self.mode == AutoEncoderModel.ENCODE:
                return mp.encode(features.features, samples, **kwargs)
            else:
                return mp.decode(features.features, samples, **kwargs)

        return self.cross_validation_aggregator(
            np.stack(
                [encode_decode_autoencode(mp, features, samples, **kwargs) for mp in self._cross_validation_models.values()],
                axis=0)
        )

    def train_predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        return self.predict(features=features, samples=samples, **kwargs)

    def init_fold(self, epoch: int, fold: int):
        pass

    def after_epoch(self, epoch: int):
        pass


class ConcatenatedMultiModel(Fittable):

    def __init__(self,
                 models: List[Fittable],
                 summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        super().__init__(None, summary_provider, forecast_provider, **kwargs)
        self.models = models

    def fit_to_df(self,
                  df: MlTypes.PatchedDataFrame,
                  fitting_parameter: FittingParameter,
                  verbose: int = 0,
                  callbacks: Optional[Callable[..., None]] = None, **kwargs) -> Tuple[MlTypes.PatchedDataFrame, MlTypes.PatchedDataFrame]:
        train_frames, test_frames = list(
            zip(*[m.fit_to_df(df, fitting_parameter, verbose, callbacks, **kwargs) for m in self.models])
        )

        return None

    def predict_of_df(self,
                      df: MlTypes.PatchedDataFrame,
                      tail: int = None,
                      samples: int = 1,
                      include_labels: bool = False, **kwargs) -> Tuple[Union[FeaturesWithLabels, FeaturesWithReconstructionTargets], MlTypes.PatchedDataFrame]:
        frames, predictions_df = list(
            [m.predict_of_df(df, tail, samples, include_labels, **kwargs) for m in self.models]
        )

        return None

class __ConcatenatedMultiModel(Model):
    # FIXME now we still need to fix the way how the features and labels get extracted by providing the kwargs to the extractor!!
    def __init__(self,
                 model_provider: Callable[..., ModelProvider],
                 features_and_labels_definition: FeaturesLabels,
                 range_arguments: List[Dict[str, Any]],
                 cross_validation_aggregator: Callable[[np.ndarray], np.ndarray] = partial(np.mean, axis=0),
                 **kwargs):
        super().__init__(features_and_labels_definition, **kwargs)
        self.model_provider = model_provider
        self.cross_validation_aggregator = cross_validation_aggregator
        self.range_arguments = range_arguments
        self._cross_validation_models: Dict[int, List[ModelProvider]] = defaultdict(
            lambda: [call_callable_dynamic_args(self.model_provider, **self.kwargs, **kwa) for kwa in range_arguments]
        )

    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        for key in self._cross_validation_models.keys():
            for m, range_kwargs in zip(self._cross_validation_models[key], self.range_arguments):
                m.init_fit(fitting_parameter, **kwargs, **range_kwargs)

    def fit_batch(self, xyw: FoldXYWeight, **kwargs):
        for m, range_kwargs in zip(self._cross_validation_models[xyw.fold], self.range_arguments):
            m.fit_batch(xyw, **kwargs, **range_kwargs)

    def after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        for m in self._cross_validation_models[fold]:
            m.after_epoch(epoch, fold_epoch, train_data, test_data)

    def finish_learning(self, **kwargs):
        for key in self._cross_validation_models.keys():
            for m, range_kwargs in zip(self._cross_validation_models[key], self.range_arguments):
                m.finish_learning(**kwargs, **range_kwargs)

    def predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        # after stacking we have one prediction per batch in the shape of [batch_dim, prediction_dim]
        aggregated = np.array([self.cross_validation_aggregator(
            np.stack(
                [mp.predict(features.features, samples, **kwargs) for mp in self._cross_validation_models.values()],
                axis=0
            )
        ) for i, _ in enumerate(self.range_arguments)])

        # the aggregated values will then have the shape [len_range_param, batch_dim, prediction_dim]
        # therefore we have to swap the axis 0 and 1 such that we get [batch_dim, len_range_param, prediction_dim]
        return aggregated.swapaxes(0, 1)

    def train_predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        # after stacking we have one prediction per batch in the shape of [batch_dim, prediction_dim]
        aggregated = np.array([self.cross_validation_aggregator(
            np.stack(
                [mp.train_predict(features.features, samples, **kwargs) for mp in self._cross_validation_models.values()],
                axis=0
            )
        ) for i, _ in enumerate(self.range_arguments)])

        # the aggregated values will then have the shape [len_range_param, batch_dim, prediction_dim]
        # therefore we have to swap the axis 0 and 1 such that we get [batch_dim, len_range_param, prediction_dim]
        return aggregated.swapaxes(0, 1)

    def calculate_train_test_loss(self, fold: int, train_data: XYWeight, test_data: List[XYWeight]) -> MlTypes.Loss:
        return (
            np.mean([m.calculate_loss(train_data) for m in self._cross_validation_models[fold]]),
            [np.mean([m.calculate_loss(td) for m in self._cross_validation_models[fold]]) for td in test_data],
        )

    def init_fold(self, epoch: int, fold: int):
        pass

    def after_epoch(self, epoch: int):
        pass


