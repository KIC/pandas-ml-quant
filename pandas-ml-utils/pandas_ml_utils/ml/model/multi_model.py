import inspect
import logging
import math
from copy import deepcopy
from functools import partial
from typing import List, Callable, Union, Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler
from ..data.splitting.sampeling.extract_multi_model_label import ExtractMultiMultiModelSampler
from ..data.extraction import FeaturesAndLabels

_log = logging.getLogger(__name__)


class MultiModel(Model):

    def __init__(self,
                 basis_model: Model,
                 nr_models: Union[int, List, Tuple],
                 model_index_variable: str = "i",
                 features_and_labels: FeaturesAndLabels = None,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(
            # we need to generate one label and sample weight for each of the sub models such that the global
            #  DataSampler returns one numpy array (which we then need to slice up again for each individual model).
            MultiModel._expand_features_and_labels(
                features_and_labels if features_and_labels is not None else basis_model.features_and_labels,
                nr_models,
                model_index_variable
            ),
            summary_provider,
            **kwargs
        )
        self.basis_model = basis_model
        self.nr_models = nr_models if isinstance(nr_models, int) else len(nr_models)
        self.sub_models: List[Model] = []

    @staticmethod
    def _expand_features_and_labels(features_and_labels, nr_of_models, model_index_variable):
        # early exit if nothing to do!
        if model_index_variable is None:
            return features_and_labels

        def wrap_partial(i, selector):
            if callable(selector):
                if model_index_variable in inspect.signature(selector).parameters.keys():
                    return partial(selector, **{model_index_variable: i})

            return selector

        models_iter = range(nr_of_models) if isinstance(nr_of_models, int) else nr_of_models
        labels = [wrap_partial(i, l) for i in models_iter for l in features_and_labels.labels]
        features_and_labels = features_and_labels.with_labels(labels)

        if features_and_labels.sample_weights is not None:
            models_iter = range(nr_of_models) if isinstance(nr_of_models, int) else nr_of_models
            weights = [wrap_partial(i, w) for i in models_iter for w in features_and_labels.sample_weights]
            features_and_labels = features_and_labels.with_sample_weights(weights)

        return features_and_labels

    def fit(self, sampler: Sampler, **kwargs) -> float:
        sub_model_losses = []

        # copy sampler and add a new cross validator extracting the model
        for i_model in range(self.nr_models):
            if "verbose" in kwargs and kwargs["verbose"]:
                print(f"fit model {i_model + 1} / {self.nr_models}")
            else:
                _log.info(f"fit model {i_model + 1} / {self.nr_models}")

            sm_loss = self.sub_models[i_model]\
                .fit(ExtractMultiMultiModelSampler(i_model, self.nr_models, sampler), **kwargs)
            sub_model_losses.append(sm_loss)

        self._history = [sm._history for sm in self.sub_models]
        return np.array(sub_model_losses).mean()

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # call each sub_model.predict()
        predictions = [m.predict_sample(x) for m in self.sub_models]

        # and then concatenate the arrays back into one prediction
        return np.stack(predictions, 1)

    def plot_loss(self, figsize=(8, 6), columns=None):
        # override the plot function and plot the loss per model and fold
        import matplotlib.pyplot as plt

        columns = self.nr_models if columns is None else int(columns)
        rows = math.ceil(self.nr_models / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))
        axes = axes.flatten()

        for m_nr, m_hist in enumerate(self._history):
            ax = axes[m_nr]

            for fold_nr, fold_loss in enumerate(m_hist):
                p = ax.plot(fold_loss[0], '-', label=f'{m_nr}: {fold_nr}: loss')
                ax.plot(fold_loss[1], '--', color=p[-1].get_color(), label=f'{m_nr}: {fold_nr}: val loss')

        plt.legend(loc='upper right')
        return fig, ax

    def __call__(self, *args, **kwargs):
        copy = deepcopy(self)
        nr_models = self.nr_models

        # get the labels and sample weights from the basis model
        l = self.features_and_labels.labels
        w = self.features_and_labels.sample_weights

        # this function is now reverse splitting up the generated labels
        # which is useful if one of the sum models gets used individually somewhere else
        def create_sub_model(i):
            # copy the FeaturesAndLabels object from the basis model
            fl = deepcopy(self.features_and_labels)

            # replace labels with one slice per model
            fl._labels = [l[i*nr_models:(i+1)*nr_models]]
            fl._sample_weights = [w[i*nr_models:(i+1)*nr_models]] if w is not None else None

            # create a new model from the basis model and replace the FeaturesAndLabels object
            sm = self.basis_model(*args, **kwargs)
            sm._features_and_labels = fl
            return sm

        # initialize all sub models and use the mutated FeaturesAndLabels objects we prepared for each model
        # Note! the data is actually engineered by the usage of the MultiModel's FeaturesAndLabels object and NOT
        # by the individual once. However this is needed to keep a FeaturesAndLabels and Models consistent.
        copy.sub_models = [create_sub_model(i) for i in range(self.nr_models)]
        return copy
