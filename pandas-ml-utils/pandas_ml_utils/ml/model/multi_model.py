import logging
from copy import deepcopy
from typing import List, Callable

import numpy as np
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels
from pandas_ml_common import Typing
from pandas_ml_utils.ml.summary import Summary
from .base_model import Model
from ..data.splitting.sampeling import Sampler

_log = logging.getLogger(__name__)


class AutoEncoderModel(Model):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 nr_models: int = 1,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.nr_models = nr_models

    def fit(self, sampler: Sampler, **kwargs) -> float:
        for i, s in enumerate(sampler.sample()):
            x, y, x_val, y_val, w, w_val = s[0][0], s[0][1], s[1][0], s[1][1], s[0][3], s[1][3]

            nr_features = x.shape[1] // self.nr_models
            nr_labels = y.shape[1] // self.nr_models

            for i in range(self.nr_models):
                _x = x[:, (nr_features * i):(nr_features * (i + 1))]
                _x_val = x_val[:, (nr_features * i):(nr_features * (i + 1))]

                _y = y[:, (nr_labels * i):(nr_labels * (i + 1))]
                _y_val = y_val[:, (nr_labels * i):(nr_labels * (i + 1))]

                if w is not None:
                    if w.ndim > 1:
                        if w.shape[1] > 1:
                            # TODO analog x and y
                            pass
                        else:
                            _w = w[i]
                            _w_val = w_val[i]
                    else:
                        _w = w
                        _w_val = w_val

                # TODO sub_model.fit_fold
                # we would need to create a new sampler and call sub_model.fit(new_sampler)
                loss = self.fit_fold(i, x, y, x_val, y_val, w, w_val, **kwargs)

        # FIXME return a loss
        return 100

    def predict_sample(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    def plot_loss(self):
        pass

    def __call__(self, *args, **kwargs):
        pass