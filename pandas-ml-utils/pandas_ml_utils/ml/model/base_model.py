import logging
import os
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable, Tuple, List, Dict, Any, Union, Type

import dill as pickle
import numpy as np
import pandas as pd

from pandas_ml_common import Typing, Sampler, LazyInit
from pandas_ml_common.sampling.sampler import XYWeight
from pandas_ml_common.utils import merge_kwargs, call_callable_dynamic_args, pd_concat
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.fitting import FittingParameter
from pandas_ml_utils.ml.forecast import Forecast
from pandas_ml_utils.ml.model.statistics import ModelFitStatistics
from pandas_ml_utils.ml.summary import Summary
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels
from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import extract_frames_for_fit, \
    extract_frames_for_predict, extract_frames_for_backtest, FeaturesWithLabels, FeaturesWithTargets, \
    FeaturesWithRequiredSamples

_log = logging.getLogger(__name__)


class Model(object):
    """
    Represents a statistical or ML model and holds the necessary information how to interpret the columns of a
    pandas *DataFrame* ( :class:`.FeaturesAndLabels` ). Currently available implementations are:
     * SkitModel - provide any skit learn classifier or regressor
     * KerasModel - provide a function returning a compiled keras model
     * MultiModel - provide a model which will copied (and fitted) for each provided target
    """

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

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[Typing.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        """
        All implementations of `Model` need to pass two arguments to `super().__init()__`.

        :param features_and_labels: the :class:`.FeaturesAndLabels` object defining all the features,
                                    feature engineerings and labels
        :param summary_provider: a summary provider in the most simple case just holds a `pd.DataFrame` containing all
                                 the labels and all the predictions and optionally loss and target values. Since
                                 constructors as callables as well it is usually enough tho just pass the type i.e.
                                 `summary_provider=BinaryClassificationSummary`
        :param kwargs:
        """
        self._features_and_labels = features_and_labels
        self._summary_provider = summary_provider
        self._forecast_provider = forecast_provider
        self._statistics = ModelFitStatistics()
        self._history = defaultdict(dict)  # todo move to stats
        self.kwargs = kwargs

        # fields holding information after fit
        self._labels_columns = None
        self._feature_columns = None
        self._fit_meta_data: FittingParameter = None

    @property
    def features_and_labels(self):
        return self._features_and_labels

    @property
    def summary_provider(self):
        return self._summary_provider

    @property
    def forecast_provider(self):
        return self._forecast_provider

    def fit_to_df(self,
            df: Typing.PatchedDataFrame,
            fitting_parameter: FittingParameter,
            type_mapping: Dict[Type, callable] = {},
            verbose: int = 0,
            callbacks=None, **kwargs) -> Tuple[FeaturesWithLabels, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        """
        The fit function is the most complex function executing all the magic, as the fit function calls the following
        sub functions:
            * init_fit
            * train_predict
            * predict
            * init_fold,
            * merge_folds,
            * finish_learning

        :param df:
        :param fitting_parameter:
        :param verbose:
        :param callbacks:
        :param kwargs:
        :return:
        """
        typemap_fitting = {SubModelFeature: lambda df, model, **kwargs: model.fit(df, **kwargs), **type_mapping}
        merged_kwargs = merge_kwargs(self.features_and_labels.kwargs, self.kwargs, kwargs)
        frames = extract_frames_for_fit(df, self.features_and_labels, typemap_fitting, fitting_parameter, verbose, **merged_kwargs)
        xyw = XYWeight(frames.features, frames.labels, frames.sample_weights)

        # remember parameters used to fit and feature and label names for later reconstruction
        self._fit_meta_data = fitting_parameter
        self._labels_columns = xyw.y.columns.tolist()
        self._feature_columns = xyw.x.columns.tolist()

        # initialize the fit of the model
        self.init_fit(**merged_kwargs)
        processed_batches = 0

        # set up a sampler for the data
        sampler = Sampler(
                xyw,
                splitter=fitting_parameter.splitter,
                filter=fitting_parameter.filter,
                cross_validation=fitting_parameter.cross_validation,
                epochs=fitting_parameter.epochs,
                fold_epochs=fitting_parameter.fold_epochs,
                batch_size=fitting_parameter.batch_size
        ).with_callbacks(
            on_start=partial(self._statistics.record_meta, fitting_param=fitting_parameter),
            on_fold=self.init_fold,
            after_fold_epoch=partial(self.after_fold_epoch, callbacks=callbacks, verbose=verbose),
            after_epoch=self.merge_folds,
            after_end=self.finish_learning
        )

        # fit the model
        for batch in sampler.sample_for_training():
            self.fit_batch(batch.x, batch.y, batch.weight, batch.fold, **merged_kwargs)
            processed_batches += 1

        if processed_batches <= 0:
            raise ValueError(f"Not enough data {[len(f) for f in sampler.frames[0]]}")

        training_data = sampler.get_in_sample_features()
        df_training_prediction = self.train_predict(training_data, **merged_kwargs)

        test_data = sampler.get_out_of_sample_features()
        df_test_prediction = self.predict(test_data) if len(test_data) > 0 else pd.DataFrame({})

        return frames, df_training_prediction, df_test_prediction

    def predict_of_df(
            self,
            df: Typing.PatchedDataFrame,
            type_mapping: Dict[Type, callable] = {},
            tail: int = None,
            samples: int = 1,
            include_labels: bool = False,
            **kwargs) -> Tuple[Union[FeaturesWithLabels, FeaturesWithTargets], Typing.PatchedDataFrame]:
        typemap_pred = {SubModelFeature: lambda df, model, **kwargs: model.predict(df, **kwargs), **type_mapping}
        merged_kwargs = merge_kwargs(self.features_and_labels.kwargs, self.kwargs, kwargs)
        if include_labels:
            frames = extract_frames_for_backtest(df, self.features_and_labels, typemap_pred, tail, **merged_kwargs)
            predictions = self.predict(frames.features, samples=samples, **kwargs)
        else:
            frames = extract_frames_for_predict(df, self.features_and_labels, typemap_pred, tail, **merged_kwargs)
            predictions = self.predict(frames.features, frames.targets, frames.latent, samples, **kwargs)

        return frames, predictions

    def after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight], verbose, callbacks, loss_history_key=None):
        train_loss, test_loss = self.calculate_train_test_loss(epoch, fold, train_data, test_data, verbose)
        self._statistics.record_loss(loss_history_key or fold, epoch, fold, fold_epoch, train_loss, test_loss)

        call_callable_dynamic_args(
            callbacks,
            epoch=epoch, fold=fold, fold_epoch=fold_epoch, loss=train_loss, test_loss=test_loss, val_loss=test_loss,
            y_train=train_data.y, y_test=[td.y for td in test_data],
            y_hat_train=LazyInit(lambda: self.predict(train_data.x)),
            y_hat_test=[LazyInit(lambda: self.predict(td.x)) for td in test_data]
        )

    def calculate_train_test_loss(self, epoch, fold, train_data: XYWeight, test_data: List[XYWeight], verbose: int):
        train_loss = self.calculate_loss(fold, train_data.x, train_data.y, train_data.weight)
        if len(test_data) > 0:
            test_loss = np.array([self.calculate_loss(fold, x, y, w) for x, y, w in test_data if len(x) > 0]).mean()
        else:
            test_loss = np.NaN

        if verbose > 0:
            print(f"fold: {fold}\tepoch: {epoch}\ttrain loss: {train_loss:.5f}, test loss: {test_loss:.5f}")

        return train_loss, test_loss

    @abstractmethod
    def init_fit(self, **kwargs):
        pass

    @abstractmethod
    def init_fold(self, epoch: int, fold: int):
        pass

    @abstractmethod
    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        raise NotImplemented

    @abstractmethod
    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        raise NotImplemented

    @abstractmethod
    def merge_folds(self, epoch: int):
        pass

    @abstractmethod
    def train_predict(self, *args, **kwargs) -> Typing.PatchedDataFrame:
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples: int = 1, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def finish_learning(self):
        pass

    def save(self, filename: str):
        """
        save model to disk
        :param filename: filename inclusive file extension
        :return: None
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

        print(f"saved model to: {os.path.abspath(filename)}")

    def plot_loss(self, figsize=(8, 6), **kwargs):
        return self._statistics.plot_loss(figsize, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        returns a copy pf the model with eventually different configuration (kwargs). This is useful for hyper paramter
        tuning or for MultiModels

        :param args:
        :param kwargs: arguments which are eventually provided by hyperopt or by different targets
        :return:
        """
        copy = deepcopy(self)
        copy.kwargs = merge_kwargs(copy.kwargs, kwargs)
        return copy


class AutoEncoderModel(Model):

    # mode constants
    AUTOENCODE = 'autoencode'
    ENCODE = 'encode'
    DECODE = 'decode'

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[Typing.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, forecast_provider, **kwargs)
        self.mode = AutoEncoderModel.AUTOENCODE

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame=None, latent: pd.DataFrame=None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        if self.mode == AutoEncoderModel.AUTOENCODE:
            return self._auto_encode(features, samples, **kwargs)
        elif self.mode == AutoEncoderModel.ENCODE:
            return self._encode(features, samples, **kwargs)
        elif self.mode == AutoEncoderModel.DECODE:
            return self._decode(latent, samples, **kwargs)
        else:
            raise ValueError("Illegal mode")

    def as_auto_encoder(self) -> 'AutoEncoderModel':
        copy = self()
        copy.mode = AutoEncoderModel.AUTOENCODE
        return copy

    def as_encoder(self) -> 'AutoEncoderModel':
        copy = self()
        copy.mode = AutoEncoderModel.ENCODE
        return copy

    def as_decoder(self) -> 'AutoEncoderModel':
        copy = self()
        copy.mode = AutoEncoderModel.DECODE
        return copy

    @abstractmethod
    def _auto_encode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def _encode(self, features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def _decode(self, latent_features: pd.DataFrame, samples, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented


class SubModelFeature(object):

    def __init__(self, name: str, model: Model):
        self.name = name
        self.model = model

    def fit(self, df: Typing.PatchedDataFrame, **kwargs):
        _log.info(f"fitting submodel: {self.name}")
        with df.model() as m:
            fit = m.fit(self.model, **kwargs)
            self.model = fit.model

        _log.info(f"fitted submodel: {fit}")
        return self.predict(df, **kwargs)

    def predict(self, df: Typing.PatchedDataFrame, **kwargs):
        if isinstance(self.model, AutoEncoderModel):
            return df.model.predict(self.model.as_encoder(), **kwargs)[PREDICTION_COLUMN_NAME]
        else:
            return df.model.predict(self.model, **kwargs)[PREDICTION_COLUMN_NAME]


class ConcatenatedMultiModel(Model):
    """
    Trains the same model configurations for a list or kwargs, i.e. to predict different time steps like t+1, t+2, t+3
    """

    def __init__(self,
                 model_provider: Callable[[], Model],
                 kwargs_list: List[Dict[str, Any]],
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[Typing.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, forecast_provider, **kwargs)
        self.kwargs_list = kwargs_list
        self.ensemble = [model_provider(features_and_labels=features_and_labels, **ka, **kwargs) for ka in kwargs_list]

    def fit_to_df(self,
            df: Typing.PatchedDataFrame,
            fitting_parameter: FittingParameter,
            type_mapping: Dict[Type, callable] = {},
            verbose: int = 0,
            callbacks=None, **kwargs) -> Tuple[FeaturesWithLabels, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        list_of_frames_df_training_prediction_df_test_predictions = \
            [m.fit_to_df(df, fitting_parameter, type_mapping, verbose, callbacks, **kwargs) for m in self.ensemble]

        # merge ensemble
        all_frames, all_train, all_test = list(zip(*list_of_frames_df_training_prediction_df_test_predictions))
        train = pd.concat(all_train, axis=1)
        test = pd.concat(all_test, axis=1)
        frames = self._reassemble_features_with_labels(all_frames)

        return frames, train, test

    def predict_of_df(
            self,
            df: Typing.PatchedDataFrame,
            type_mapping: Dict[Type, callable] = {},
            tail: int = None,
            samples: int = 1,
            include_labels: bool = False,
            **kwargs) -> Tuple[Union[FeaturesWithLabels, FeaturesWithTargets], Typing.PatchedDataFrame]:
        # for each model in ensemble
        lof_of_frames_with_predictions = \
            [m.predict_of_df(df, type_mapping, tail, samples, include_labels, **kwargs) for m in self.ensemble]

        all_frames, all_predictions = list(zip(*lof_of_frames_with_predictions))
        frames = self._reassemble_features_with_labels(all_frames) if include_labels else self._reassemble_features(all_frames)
        predictions = pd.concat(all_predictions, axis=1)

        return frames, predictions

    @abstractmethod
    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        losses = np.array([m.calculate_loss(fold, x, y_true, weight) for m in self.ensemble])
        return losses.sum()

    def _reassemble_features_with_labels(self, all_frames: List[FeaturesWithLabels]):
        return FeaturesWithLabels(
            FeaturesWithRequiredSamples(
                all_frames[0].features,
                max([f.features_with_required_samples.min_required_samples for f in all_frames]),
                all_frames[0].features_with_required_samples.nr_of_features
            ),
            pd_concat([f.labels for f in all_frames if f.labels is not None], axis=1),
            None,  # we don't support auto encoder ensembles at this time
            pd_concat([f.targets for f in all_frames if f.targets is not None], axis=1),
            pd_concat([f.sample_weights for f in all_frames if f.sample_weights is not None], axis=1),
            pd_concat([f.gross_loss for f in all_frames if f.gross_loss is not None], axis=1),
        )

    def _reassemble_features(self, all_frames: List[FeaturesWithTargets]):
        return FeaturesWithTargets(
            pd_concat([f.features for f in all_frames if f.features is not None], axis=1),
            pd_concat([f.targets for f in all_frames if f.targets is not None], axis=1),
            None,  # we don't support auto encoder ensembles at this time
        )
