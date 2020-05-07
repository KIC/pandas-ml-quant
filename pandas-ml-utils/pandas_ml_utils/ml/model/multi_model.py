import logging
from copy import deepcopy
from typing import List, Callable, Generator, Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler

_log = logging.getLogger(__name__)


# we need to create a fake sampler to make this thing work
class _Sampler(object):

    def __init__(self, train: List[np.ndarray], test: List[np.ndarray]):
        self.train = train
        self.test = test

    def sample(self) -> Generator[Tuple[List[np.ndarray], List[np.ndarray]], None, None]:
        yield self.train, self.test


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

        for i, s in enumerate(sampler.sample()):
            x, y, x_val, y_val, t, t_val, w, w_val = s[0][0], s[0][1], s[1][0], s[1][1], s[0][2], s[1][2], s[0][3], s[1][3]
            nr_labels = y.shape[1] // self.nr_models

            def cut(arr, i):
                return arr[:, (nr_labels * i):(nr_labels * (i + 1))]

            for i in range(self.nr_models):
                _y, _y_val = cut(y, i), cut(y_val, i)
                _w, _w_val = None, None
                _t, _t_val = t, t_val

                if w is not None:
                    if w.ndim > 1:
                        if w.shape[1] > 1:
                            _w = cut(w, i)
                            _w_val = cut(w_val, i)
                        else:
                            _w = w[i]
                            _w_val = w_val[i]
                    else:
                        _w = w
                        _w_val = w_val

                # now call sub_model.fit() with new a new sampler for each label
                loss = self.sub_models[i].fit(_Sampler([x, _y, _t, _w], [x_val, _y_val, _t_val, _w_val]))
                sub_model_losses.append(loss)

        self._history = [sm._history for sm in self.sub_models]
        return np.array(sub_model_losses).mean()

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # call each sub_model.predict()
        predictions = [m.predict_sample(x) for m in self.sub_models]

        # and then concatenate the arrays back into one prediction
        return np.stack(predictions, 1)

    def plot_loss(self, figsize=(8, 6)):
        # override the plot function and plot the loss per model and fold
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(figsize if figsize else plt.rcParams.get('figure.figsize')))

        for m_nr, m_hist in enumerate(self._history):
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
