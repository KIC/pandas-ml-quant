import inspect
import logging
import math
from copy import deepcopy
from functools import partial
from typing import List, Callable, Union, Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import call_callable_dynamic_args
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler
from ..data.splitting.sampeling.extract_multi_model_label import ExtractMultiMultiModelSampler
from ..data.extraction import FeaturesAndLabels

_log = logging.getLogger(__name__)


class MultiModel(Model):

    def __init__(self,
                 basis_model: Model,
                 model_args: Union[int, List, Tuple],
                 model_index_variable: str = "i",
                 features_and_labels: FeaturesAndLabels = None,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(
            # we need to generate one label and sample weight for each of the sub models such that the global
            #  DataSampler returns one numpy array (which we then need to slice up again for each individual model).
            MultiModel._expand_features_and_labels(
                features_and_labels if features_and_labels is not None else basis_model.features_and_labels,
                MultiModel.models_iter(model_args),
                model_index_variable
            ),
            summary_provider,
            **kwargs
        )
        self.basis_model = basis_model
        self.sub_models: List[Model] = []
        self.model_index_variable = model_index_variable
        self.model_args = model_args
        self.nr_models = len(list(MultiModel.models_iter(model_args)))

    @staticmethod
    def _expand_features_and_labels(features_and_labels, model_args_iter, model_index_variable):
        # remove the variable which controls the avirous aspects of each of the multi model
        # we replace this information by the usage of a partial
        if model_index_variable in features_and_labels.kwargs:
            del features_and_labels.kwargs[model_index_variable]

        # early exit if nothing to do!
        if model_index_variable is None:
            return features_and_labels

        def wrap_partial(i, selector):
            if callable(selector):
                if model_index_variable in inspect.signature(selector).parameters.keys():
                    return partial(selector, **{model_index_variable: i})

            return selector

        labels = [wrap_partial(i, l) for i in model_args_iter for l in features_and_labels.labels if callable(l)]
        features_and_labels = features_and_labels.with_labels(labels)
        features_and_labels.set_label_columns(None, True)

        if features_and_labels.sample_weights is not None:
            weights = [wrap_partial(i, w) for i in model_args_iter for w in features_and_labels.sample_weights if callable(w)]
            features_and_labels = features_and_labels.with_sample_weights(weights)

        if features_and_labels.gross_loss is not None:
            gross_loss = [wrap_partial(i, gl) for i in model_args_iter for gl in features_and_labels.gross_loss if callable(gl)]
            features_and_labels = features_and_labels.with_gross_loss(gross_loss)

        return features_and_labels

    @staticmethod
    def models_iter(nr_of_models):
        return range(nr_of_models) if isinstance(nr_of_models, int) else nr_of_models

    def fit(self, sampler: Sampler, **kwargs) -> float:
        sub_model_losses = []
        nr_of_models = self.nr_models

        # copy sampler and add a new cross validator extracting the model
        for i_model, arg in enumerate(MultiModel.models_iter(self.model_args)):
            if "verbose" in kwargs and kwargs["verbose"]:
                print(f"fit model {i_model + 1} / {nr_of_models}")
            else:
                _log.info(f"fit model {i_model + 1} / {nr_of_models}")

            if "on_model" in kwargs:
                for callback in kwargs["on_model"]:
                    callback(i_model, {**kwargs, self.model_index_variable: arg})

            sm_loss = self.sub_models[i_model]\
                .fit(ExtractMultiMultiModelSampler(i_model, nr_of_models, sampler), **deepcopy(kwargs))
            sub_model_losses.append(sm_loss)

        self._history = [sm._history for sm in self.sub_models]
        return np.array(sub_model_losses).mean()

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # call each sub_model.predict()
        predictions = [m.predict_sample(x) for m in self.sub_models]

        # and then concatenate the arrays back into one prediction
        return np.concatenate(predictions, 1) if predictions[0].ndim > 1 else np.stack(predictions, 1)

    def plot_loss(self, figsize=(8, 6), columns=None):
        # override the plot function and plot the loss per model and fold
        import matplotlib.pyplot as plt
        nr_of_models = self.nr_models

        columns = nr_of_models if columns is None else int(columns)
        rows = math.ceil(nr_of_models / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))
        axes = axes.flatten()

        for m_nr, m_hist in enumerate(self._history):
            ax = axes[m_nr]

            for fold_nr, fold_loss in enumerate(m_hist):
                if len(fold_loss[0]) <= 1:
                    _log.warning("can not plot single epoch loss")

                p = ax.plot(fold_loss[0], '-', label=f'{m_nr}: {fold_nr}: loss')
                ax.plot(fold_loss[1], '--', color=p[-1].get_color(), label=f'{m_nr}: {fold_nr}: val loss')

        plt.legend(loc='upper right')
        return fig, axes

    def _slices(self, i, total_nr_of_columns=1) -> slice:
        nr_labels = total_nr_of_columns // self.nr_models
        return slice(i * nr_labels, (i + 1) * nr_labels)

    def __call__(self, *args, **kwargs):
        copy = deepcopy(self)
        nr_of_models = self.nr_models

        # get the labels and sample weights from the basis model
        l = self.features_and_labels.labels
        t = self.features_and_labels.targets
        w = self.features_and_labels.sample_weights
        gl = self.features_and_labels.gross_loss

        # this function is now reverse splitting up the generated labels
        # which is useful if one of the sum models gets used individually somewhere else
        def create_sub_model(i):
            # copy the FeaturesAndLabels object from the basis model
            fl = deepcopy(self.features_and_labels)

            # replace labels with one slice per model
            fl._labels = l[self._slices(i, len(l))]
            fl._targets = t[self._slices(i, len(t))] if t is not None else None
            fl._sample_weights = w[self._slices(i, len(w))] if w is not None else None
            fl._gross_loss = gl[self._slices(i, len(gl))] if gl is not None else None

            # create a new model from the basis model and replace the FeaturesAndLabels object
            sm = self.basis_model(*args, **kwargs)
            sm._features_and_labels = fl
            return sm

        # initialize all sub models and use the mutated FeaturesAndLabels objects we prepared for each model
        # Note! the data is actually engineered by the usage of the MultiModel's FeaturesAndLabels object and NOT
        # by the individual once. However this is needed to keep a FeaturesAndLabels and Models consistent.
        copy.sub_models = [create_sub_model(i) for i in range(nr_of_models)]
        return copy
