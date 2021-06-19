import logging
import os
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Callable, Tuple, List

import dill as pickle
import numpy as np
import pandas as pd

from pandas_ml_common import Typing, Sampler, LazyInit
from pandas_ml_common.sampling.sampler import XYWeight
from pandas_ml_common.utils import merge_kwargs, call_callable_dynamic_args
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.fitting import FittingParameter
from pandas_ml_utils.ml.forecast import Forecast
from pandas_ml_utils.ml.summary import Summary

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
        self._history = defaultdict(dict)
        self._labels_columns = None
        self._feature_columns = None
        self._fit_meta_data: FittingParameter = None
        self.kwargs = kwargs

    @property
    def features_and_labels(self):
        return self._features_and_labels

    @property
    def summary_provider(self):
        return self._summary_provider

    @property
    def forecast_provider(self):
        return self._forecast_provider

    def fit(self, x_y_weights: XYWeight, fitting_parameter: FittingParameter, verbose: int = 0, callbacks=None, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        self._fit_meta_data = fitting_parameter
        self.init_fit(**kwargs)
        processed_batches = 0

        sampler = self._sampler_with_callbacks(
            Sampler(
                x_y_weights,
                splitter=fitting_parameter.splitter,
                filter=fitting_parameter.filter,
                cross_validation=fitting_parameter.cross_validation,
                epochs=fitting_parameter.epochs,
                fold_epochs=fitting_parameter.fold_epochs,
                batch_size=fitting_parameter.batch_size
            ),
            verbose,
            callbacks
        )

        for batch in sampler.sample_for_training():
            self.fit_batch(batch.x, batch.y, batch.weight, batch.fold, **kwargs)
            processed_batches += 1

        if processed_batches <= 0:
            raise ValueError(f"Not enough data {[len(f) for f in sampler.frames[0]]}")

        training_data = sampler.get_in_sample_features()
        df_training_prediction = self.train_predict(training_data, **kwargs)

        test_data = sampler.get_out_of_sample_features()
        df_test_prediction = self.predict(test_data) if len(test_data) > 0 else pd.DataFrame({})

        return df_training_prediction, df_test_prediction

    def _sampler_with_callbacks(self, sampler: Sampler, verbose: int = 0, callbacks=None) -> Sampler:
        return sampler.with_callbacks(
            on_start=self._record_meta,
            on_fold=self.init_fold,
            after_fold_epoch=partial(self._record_loss, callbacks=callbacks, verbose=verbose),
            after_epoch=self.merge_folds,
            after_end=self.finish_learning
        )

    def _record_meta(self, epochs, batch_size, fold_epochs, cross_validation, features, labels: List[str]):
        partial_fit = any([size > 1 for size in [epochs, batch_size, fold_epochs] if size is not None])
        self._labels_columns = labels
        self._feature_columns = features

    def _record_loss(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight], verbose, callbacks, loss_history_key=None):
        train_loss = self.calculate_loss(fold, train_data.x, train_data.y, train_data.weight)
        self._history["train", loss_history_key or fold][(epoch, fold_epoch)] = train_loss

        if len(test_data) > 0:
            test_loss = np.array([self.calculate_loss(fold, x, y, w) for x, y, w in test_data if len(x) > 0]).mean()
        else:
            test_loss = np.NaN
        self._history["test", loss_history_key or fold][(epoch, fold_epoch)] = test_loss

        self.after_fold_epoch(epoch, fold, fold_epoch, train_loss, test_loss)
        if verbose > 0:
            print(f"epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}")

        call_callable_dynamic_args(
            callbacks,
            epoch=epoch, fold=fold, fold_epoch=fold_epoch, loss=train_loss, val_loss=test_loss,
            y_train=train_data.y, y_test=[td.y for td in test_data],
            y_hat_train=LazyInit(lambda: self.predict(train_data.x)),
            y_hat_test=[LazyInit(lambda: self.predict(td.x)) for td in test_data]
        )

    def init_fit(self, **kwargs):
        pass

    def init_fold(self, epoch: int, fold: int):
        pass

    @abstractmethod
    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        raise NotImplemented

    def after_fold_epoch(self, epoch, fold, fold_epoch, loss, val_loss):
        pass

    @abstractmethod
    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        raise NotImplemented

    def merge_folds(self, epoch: int):
        pass

    def train_predict(self, *args, **kwargs) -> Typing.PatchedDataFrame:
        return self.predict(*args, **kwargs)

    @abstractmethod
    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples: int = 1, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

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
        """
        plot a diagram of the training and validation losses per fold
        :return: figure and axis
        """

        import matplotlib.pyplot as plt
        cm = 'tab20c'  # 'Pastel1'
        df = pd.DataFrame(self._history)
        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))
        df['test'].plot(style='--', colormap=cm, ax=ax)
        df['train'].plot(colormap=cm, ax=ax)
        plt.legend(loc='upper right')
        return fig

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

