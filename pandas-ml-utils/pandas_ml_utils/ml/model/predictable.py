import logging
import os
from abc import ABCMeta
from abc import abstractmethod
from typing import Callable, Optional, Tuple, List, Any
from typing import Union

import dill as pickle
import numpy as np

from pandas_ml_common import MlTypes, FeaturesLabels, call_callable_dynamic_args
from pandas_ml_common.preprocessing.features_labels import FeaturesWithLabels, FeaturesWithReconstructionTargets
from pandas_ml_common.utils import merge_kwargs, to_pandas
from ..data.reconstruction import assemble_result_frame
from ..forecast import Forecast
from ...constants import PREDICTION_COLUMN_NAME

_log = logging.getLogger(__name__)


class Model(metaclass=ABCMeta):

    def __init__(self,
                 features_and_labels_definition: FeaturesLabels,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        self._features_and_labels_definition = features_and_labels_definition
        self._forecast_provider = forecast_provider
        self.kwargs = kwargs
        self.min_required_samples = None
        self._label_names = []

    @property
    def features_and_labels_definition(self) -> FeaturesLabels:
        return self._features_and_labels_definition

    @property
    def forecast_provider(self) -> Optional[Callable[[MlTypes.PatchedDataFrame], Forecast]]:
        return self._forecast_provider

    @property
    def label_names(self) -> List[Any]:
        return self._label_names

    @staticmethod
    def load(filename: str):
        """
        Loads a previously saved model from disk. By default `dill <https://pypi.org/project/dill/>`_ is used to
        serialize / deserialize a model.

        :param filename: filename of the serialized model inclusive file extension
        :return: returns a deserialized model
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
            if isinstance(model, Model):
                return model
            else:
                raise ValueError("Deserialized pickle was not a Model!")

    def forecast(self,
                 df: MlTypes.PatchedDataFrame,
                 tail: int = None,
                 samples: int = 1,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                 include_labels: bool = False,
                 **kwargs) -> Union[MlTypes.PatchedDataFrame, Forecast]:
        frames, predictions = self.predict_of_df(df, tail, samples, include_labels, **kwargs)
        fc_provider = forecast_provider or self.forecast_provider
        res_df = assemble_result_frame(
            predictions,
            (frames if isinstance(frames, FeaturesWithReconstructionTargets) else frames.features_with_required_samples).reconstruction_targets,
            frames.labels_with_sample_weights.joint_label_frame if include_labels else None,
            frames.labels_with_sample_weights.gross_loss if include_labels else None,
            frames.labels_with_sample_weights.joint_sample_weights_frame if include_labels else None,
            (frames if isinstance(frames, FeaturesWithReconstructionTargets) else frames.features_with_required_samples).joint_feature_frame
        )

        return res_df if fc_provider is None else call_callable_dynamic_args(fc_provider, df=res_df, **kwargs)

    def predict_of_df(
            self,
            df: MlTypes.PatchedDataFrame,
            tail: int = None,
            samples: int = 1,
            include_labels: bool = False,
            **kwargs) -> Tuple[Union[FeaturesWithLabels, FeaturesWithReconstructionTargets], MlTypes.PatchedDataFrame]:
        typemap_pred = {SubModelFeature: lambda df, model, **kwargs: model.predict(df, **kwargs)}
        merged_kwargs = merge_kwargs(self.kwargs, kwargs)
        extractor = df.ML.extract(self.features_and_labels_definition, typemap_pred, **merged_kwargs)
        tail = (tail + self.min_required_samples) if self.min_required_samples is not None and tail is not None else None

        if include_labels:
            frames = extractor.extract_features_labels_weights(tail=tail)
            predictions = self.predict(frames.features_with_required_samples, samples=samples, **merged_kwargs)
        else:
            frames = extractor.extract_features(tail=tail)
            predictions = self.predict(frames, samples=samples, **merged_kwargs)

        predictions_df = to_pandas(
            predictions,
            frames.common_features_index if isinstance(frames, FeaturesWithLabels) else frames.common_index,
            self.label_names
        )

        return frames, predictions_df

    @abstractmethod  # FIXME one model can predict multiple labels !!   -> List[np.ndarray]:
    def predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    def save(self, filename: str):
        """
        save model to disk
        :param filename: filename inclusive file extension
        :return: None
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

        print(f"saved model to: {os.path.abspath(filename)}")


class SubModelFeature(object):

    def __init__(self, name: str, model: Model):
        self.name = name
        self.model = model

    def fit(self, df: MlTypes.PatchedDataFrame, **kwargs):
        _log.info(f"fitting submodel: {self.name}")
        with df.model() as m:
            fit = m.fit(self.model, **kwargs)
            self.model = fit.model

        _log.info(f"fitted submodel: {fit}")
        return self.predict(df, **kwargs)

    def predict(self, df: MlTypes.PatchedDataFrame, **kwargs):
        if hasattr(self.model, 'as_encoder'):
            return df.model.predict(self.model.as_encoder(), **kwargs)[PREDICTION_COLUMN_NAME]
        else:
            return df.model.predict(self.model, **kwargs)[PREDICTION_COLUMN_NAME]
