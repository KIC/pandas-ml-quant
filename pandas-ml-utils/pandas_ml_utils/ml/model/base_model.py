import os
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Tuple, Iterable, Generator, Union, List, NamedTuple
from collections import defaultdict

import dill as pickle
import numpy as np
import pandas as pd

from pandas_ml_common import Typing, Sampler
from pandas_ml_common.sampling.sampler import XYWeight
from pandas_ml_common.utils import to_pandas, unique_level
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary


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
        self._history = defaultdict(dict)
        self._labels_columns = None
        self._fit_meta_data: Model.MetaFit = None
        self.kwargs = kwargs

    @property
    def features_and_labels(self):
        return self._features_and_labels

    @property
    def summary_provider(self):
        return self._summary_provider

    # TODO im normalen Modell und after_fold im rolling model. nach jedem merge folds darf halt nur noch ein Modell im Speicher sein.
    def fit(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        sampler = sampler.with_callbacks(
            on_start=self._record_meta,
            on_fold=self.init_fold,
            after_fold_epoch=self._record_loss,
            after_epoch=self.merge_folds,
            after_end=self.finish_learning
        )

        for batch in sampler.sample_for_training():
            self.fit_batch(batch.x, batch.y, batch.weight, **kwargs)

        training_data = sampler.get_in_sample_features()
        df_training_prediction = self.predict(training_data, **kwargs)

        test_data = sampler.get_out_of_sample_features()
        df_test_prediction = self.predict(test_data) if len(test_data) > 0 else pd.DataFrame({})

        return df_training_prediction, df_test_prediction

    def _record_meta(self, epochs, batch_size, fold_epochs, cross_validation, labels: List[str]):
        self._labels_columns = labels
        self._fit_meta_data = _MetaFit(
            epochs, batch_size, fold_epochs,
            cross_validation, any([size > 1 for size in [epochs, batch_size, fold_epochs] if size is not None])
        )

    def _record_loss(self, epoch, fold, fold_epoch, train_data: List[XYWeight], test_data: List[XYWeight]):
        train_loss = self.calculate_loss(fold, train_data.x, train_data.y, train_data.weight)
        test_loss = np.array([self.calculate_loss(fold, x, y, w) for x, y, w in test_data]).mean()
        self._history["train", fold][(epoch, fold_epoch)] = train_loss
        self._history["test", fold][(epoch, fold_epoch)] = test_loss

    @abstractmethod
    def init_fold(self, epoch: int, fold: int):
        raise NotImplemented

    @abstractmethod
    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, **kwargs):
        raise NotImplemented

    @abstractmethod
    def calculate_loss(self, fold, y_true, y_pred, weight) -> float:
        raise NotImplemented

    @abstractmethod
    def merge_folds(self, epoch: int):
        raise NotImplemented

    @abstractmethod
    def predict(self, features: pd.DataFrame, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def finish_learning(self):
        raise NotImplemented

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

        df = pd.DataFrame(self._history)
        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))

        # TODO plot train and test data ...
        for fold in unique_level(self._history.columns, 0):
            if fold < 0:
                p = ax.plot(self._history[(fold, 0)], '-', label='model (train)')
                ax.plot(self._history[(fold, 1)], '--', color=p[-1].get_color(), label='model (test)')
            else:
                p = ax.plot(self._history[(fold, 0)], '-', label=f'fold {fold} (train)')
                ax.plot(self._history[(fold, 1)], '--', color=p[-1].get_color(), label=f'fold {fold} (test)')

        plt.legend(loc='upper right')
        return fig, ax

    def __str__(self):
        try:
            import matplotlib
            matplotlib.use('module://drawilleplot')
            # FIXME plot loss

        except:
            return repr(self)

    def __call__(self, *args, **kwargs):
        """
        returns a copy pf the model with eventually different configuration (kwargs). This is useful for hyper paramter
        tuning or for MultiModels

        :param args:
        :param kwargs: arguments which are eventually provided by hyperopt or by different targets
        :return:
        """
        if not kwargs:
            return deepcopy(self)
        else:
            raise ValueError(f"construction of model with new parameters is not supported\n{type(self)}: {kwargs}")


class AutoEncoderModel(Model):

    # mode constants
    AUTOENCODE = 'autoencode'
    ENCODE = 'encode'
    DECODE = 'decode'

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.mode = AutoEncoderModel.AUTOENCODE

    def _predict(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        if self.mode == AutoEncoderModel.AUTOENCODE:
            return self._auto_encode(sampler, **kwargs)
        elif self.mode == AutoEncoderModel.ENCODE:
            return self._encode(sampler, **kwargs)
        elif self.mode == AutoEncoderModel.DECODE:
            return self._decode(sampler, **kwargs)
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
    def _auto_encode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def _encode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented

    @abstractmethod
    def _decode(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented


class _MetaFit(NamedTuple):
    epochs: int
    batch_size: int
    fold_epochs: int
    cross_validation: bool
    partial_fit: bool


