import os
from abc import abstractmethod
from copy import deepcopy
from typing import Callable, Tuple, Iterable

import dill as pickle
import numpy as np
import pandas as pd

from pandas_ml_common import Typing, Sampler, NumpySampler
from pandas_ml_common.utils import to_pandas, unique_level
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.summary import Summary


class SamplerFrameConstants(object):
    FEATURES = 0
    LABELS = 1
    TARGETS = 2
    WEIGHTS = 3
    GROSS_LOSS = 4
    LATENT = 5


class _Model(object):

    @abstractmethod
    def _fit(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        raise NotImplemented

    @abstractmethod
    def _predict(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        raise NotImplemented


class Model(_Model):
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
        self._history = []
        self.kwargs = kwargs

    @property
    def features_and_labels(self):
        return self._features_and_labels

    @property
    def summary_provider(self):
        return self._summary_provider

    def fit(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        self._history, training_prediction, test_prediction = self._fit(sampler, **kwargs)
        return training_prediction, test_prediction

    def predict(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        return self._predict(sampler)

    def __getitem__(self, item):
        """
        returns arguments which are stored in the kwargs filed. By providing a tuple, a default in case of missing
        key can be specified
        :param item: name of the item im the kwargs dict or tuple of name, default
        :return: item or default
        """
        if isinstance(item, tuple) and len(item) == 2:
            return self.kwargs[item[0]] if item[0] in self.kwargs else item[1]
        else:
            return self.kwargs[item] if item in self.kwargs else None

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
        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))
        for fold in unique_level(self._history.columns, 0):
            if fold < 0:
                p = ax.plot(self._history[(fold, 0)], '-', label='model (train)')
                ax.plot(self._history[(fold, 1)], '--', color=p[-1].get_color(), label='model (test)')
            else:
                p = ax.plot(self._history[(fold, 0)], '-', label=f'fold {fold} (train)')
                ax.plot(self._history[(fold, 1)], '--', color=p[-1].get_color(), label=f'fold {fold} (test)')

        plt.legend(loc='upper right')
        return fig, ax

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


class _NumpyModelFit(object):

    def __init__(self, **kwargs):
        self.labels_columns = None

    def _fit_with_numpy(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        # remember the label column names for reconstruction
        self.labels_columns = sampler.frames[SamplerFrameConstants.LABELS].columns.tolist()

        # wrap sampler into a numpy sampler and fit each epoch
        nr_epochs = sampler.epochs
        last_epoch = (nr_epochs - 1) if nr_epochs is not None else float('inf')
        nr_folds = sampler.nr_of_cross_validation_folds
        numpy_sampler = NumpySampler(sampler)

        # init result holder data structures
        losses = {(fold, train_test): [] for fold in range(-1, nr_folds) for train_test in range(2)}
        test_predictions, train_predictions = [], []

        for epoch, fold, train_idx, train, test_idx, test in numpy_sampler.sample_cross_validation():
            if fold < 0 < nr_folds:
                # merge a cross validated model
                train_loss, test_loss = self._fold_epoch(train, test, nr_epochs, **kwargs)
            else:
                # train one fold of an eventually cross validation model
                train_loss, test_loss = self._fit_epoch_fold(fold, train, test, nr_folds, nr_epochs, **kwargs)

            # append losses
            if isinstance(train_loss, Iterable):
                losses[(fold, 0)].extend(train_loss)
                losses[(fold, 1)].extend(test_loss)
            else:
                losses[(fold, 0)].append(train_loss)
                losses[(fold, 1)].append(test_loss)

            # fix length of losses
            if fold < 0:
                max_len = max([len(v) for v in losses.values()])
                for k, v in losses.items():
                    v.extend([np.nan] * (max_len - len(v)))

            # assemble history data frame when done
            if epoch >= last_epoch and fold < 0:
                train_predictions.append(to_pandas(self._predict_epoch(train[0]), train_idx, self.labels_columns))
                if len(test_idx) > 0:
                    test_predictions.append(to_pandas(self._predict_epoch(test[0]), test_idx, self.labels_columns))

        # reconstruct pandas data frames
        df_losses = pd.DataFrame(losses, columns=pd.MultiIndex.from_tuples(losses.keys()))
        df_train_prediction = pd.concat(train_predictions, axis=0)
        df_test_prediction = pd.concat(test_predictions, axis=0) if len(test_predictions) > 0 else pd.DataFrame({})

        return df_losses, df_train_prediction, df_test_prediction

    @abstractmethod
    def _fold_epoch(self, train, test, nr_epochs, **kwargs) -> Tuple[float, float]:
        raise NotImplemented

    @abstractmethod
    def _fit_epoch_fold(self, fold, train, test, nr_of_folds, nr_epochs, **kwargs) -> Tuple[float, float]:
        raise NotImplemented

    @abstractmethod
    def _predict_epoch(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplemented


class NumpyModel(Model, _NumpyModelFit):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.labels_columns = None

    def _fit(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        return super()._fit_with_numpy(sampler, **kwargs)

    def _predict(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        ns = NumpySampler(sampler)
        prediction = np.array([self._predict_epoch(t[0], **kwargs) for (_, t, _) in ns.sample_full_epochs()]).swapaxes(0, 1)
        return to_pandas(prediction, sampler.frames[SamplerFrameConstants.FEATURES].index, self.labels_columns)


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


