import math
from typing import Tuple, Callable, Any

import numpy as np
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import roc_curve, auc

from pandas_ml_common import Typing
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import html
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.model.base_model import Model

from pandas_ml_utils.ml.summary import Summary


class ReconstructionSummary(Summary):

    def __init__(self,
                 df: Typing.PatchedDataFrame,
                 model: Model,
                 reconstruction_plotter: Callable[[Axis, np.ndarray], Any] = lambda ax, x: ax.plot(x),
                 **kwargs):
        super().__init__(df, model, **kwargs)
        self.reconstruction_plotter = reconstruction_plotter

    def reconstruction_plot(self):
        import matplotlib.pyplot as plt

        _y = self.df[LABEL_COLUMN_NAME]._.values
        _y_hat = self.df[PREDICTION_COLUMN_NAME]._.values
        fig, ax = plt.subplots(2, 5, figsize=(20, 10))
        ax = ax.flatten()

        # random samples
        indices = np.random.randint(_y.shape[0], size=10)
        _y = _y[indices, :]
        _y_hat = _y_hat[indices, :]

        # print shape
        print(_y.shape, _y_hat.shape)

        for i in range(10):
            self.reconstruction_plotter(ax[i], _y[i])
            self.reconstruction_plotter(ax[i], _y_hat[i])

            ax[i].legend(["original", "reconstruct"])

        return fig

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(
            fit=self,
            recon_plot=plot_to_html_img(self.reconstruction_plot)
        )
