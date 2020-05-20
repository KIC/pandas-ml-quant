import os
from copy import deepcopy
from typing import Callable, Tuple

import dill as pickle
import numpy as np

from pandas_ml_common import Typing
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_utils.ml.data.splitting.sampeling import Sampler
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
                                 constructors as callables as well it is usually enoug tho just pass the type i.e.
                                 `summary_provider=BinaryClassificationSummary`
        :param kwargs:
        """
        self._features_and_labels = features_and_labels
        self._summary_provider = summary_provider
        self._validation_indices = []
        self._history = []
        self.kwargs = kwargs

    @property
    def features_and_labels(self):
        return self._features_and_labels

    @property
    def summary_provider(self):
        return self._summary_provider

    @property
    def validation_indices(self):
        return self._validation_indices

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

    def fit(self, sampler: Sampler, **kwargs) -> float:
        """
        draws folds from the data generator as long as it yields new data and fits the model to one fold

        :param sampler: a data generating process class:`pandas_ml_utils.ml.data.splitting.sampeling.Sampler`
        :return: returns the average loss over oll folds
        """

        # sample: train[features, labels, target, weights], test[features, labels, target, weights]
        losses = [self.fit_fold(i, s[0][0], s[0][1], s[1][0], s[1][1], s[0][3], s[1][3], **kwargs)
                  for i, s in enumerate(sampler.sample())]

        self._history = losses

        # this loss is used for hyper parameter tuning so we take the average of the minimum loss of each fold
        return np.array([(fold_loss[0].min() if fold_loss[0].size > 0 else np.nan) for fold_loss in losses]).mean()

    def plot_loss(self, figsize=(8, 6), secondary_y=False, **kwargs):
        """
        plot a diagram of the training and validation losses per fold
        :return: figure and axis
        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))

        for fold_nr, fold_loss in enumerate(self._history):
            p = ax.plot(fold_loss[0], '-', label=f'{fold_nr}: loss')
            ax2 = ax.twinx() if secondary_y else ax
            ax2.plot(fold_loss[1], '--', color=p[-1].get_color(), label=f'{fold_nr}: val loss')

        plt.legend(loc='upper right')
        return fig, ax

    def predict(self, sampler: Sampler, **kwargs) -> np.ndarray:
        """
        predict as many samples as we can sample from the sampler

        :param sampler:
        :return:
        """
        # make shape (rows, samples, ...)
        return np.array([self.predict_sample(t[0]) for (t, _) in sampler.sample()]).swapaxes(0, 1)

    def fit_fold(self,
                 fold_nr: int,
                 x: np.ndarray, y: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray,
                 sample_weight_train: np.ndarray, sample_weight_test: np.ndarray,
                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        function called to fit the model to one fold of the data generator (i.e. k-folds)
        :param fold_nr: number of fold in case of cross validation is used
        :param x: x
        :param y: y
        :param x_val: x validation
        :param y_val: y validation
        :param sample_weight_train: sample weights for loss penalisation (default np.ones)
        :param sample_weight_test: sample weights for loss penalisation (default np.ones)
        :return: loss of the fit
        """
        pass

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        prediction of the model for each sample and target

        :param x: x
        :return: prediction of the model for each target
        """

        pass

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

