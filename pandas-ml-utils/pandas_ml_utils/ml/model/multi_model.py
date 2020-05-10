import logging
import math
from copy import deepcopy
from typing import List, Callable

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler
from ..data.splitting.sampeling.extract_multi_model_label import ExtractMultiMultiModelSampler

_log = logging.getLogger(__name__)


class MultiModel(Model):

    def __init__(self,
                 basis_model: Model,
                 nr_models: int,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(basis_model.features_and_labels, summary_provider, **kwargs)
        self.basis_model = basis_model
        self.nr_models = nr_models
        self.sub_models: List[Model] = []

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
        
        def create_sub_model(i):
            l = self.features_and_labels.labels
            w = self.features_and_labels.sample_weights

            fl = deepcopy(self.features_and_labels)
            fl._labels = [l[i*nr_models:(i+1)*nr_models]]
            fl._sample_weights = [w[i*nr_models:(i+1)*nr_models]] if w is not None else None

            sm = self.basis_model(*args, **kwargs)
            sm._features_and_labels = fl
            return sm

        copy.sub_models = [create_sub_model(i) for i in range(self.nr_models)]
        return copy
