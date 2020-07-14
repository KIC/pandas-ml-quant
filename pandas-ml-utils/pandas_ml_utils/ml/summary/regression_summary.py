from typing import Tuple

from matplotlib.axis import Axis
from matplotlib.figure import Figure

from pandas_ml_common import Typing
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import html
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import Summary
from pandas_ml_utils.ml.model.base_model import Model


class RegressionSummary(Summary):

    def __init__(self, df: Typing.PatchedDataFrame, model: Model, true_columns=None, pred_columns=None, **kwargs):
        super().__init__(df, model, **kwargs)
        self.true_columns = LABEL_COLUMN_NAME if true_columns is None else true_columns
        self.pred_columns = PREDICTION_COLUMN_NAME if pred_columns is None else pred_columns

    def plot_true_pred_scatter(self, figsize=(6, 6), alpha=0.1) -> Tuple[Figure, Axis]:
        import matplotlib.pyplot as plt

        df = self.df
        x = df[self.pred_columns]._.values
        y = df[self.true_columns]._.values
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())

        fig, axis = plt.subplots(figsize=figsize)
        plt.axes(aspect='equal')
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.scatter(x, y, alpha=alpha)
        plt.xlim([mn, mx])
        plt.ylim([mn, mx])
        plt.plot([mn, mx], [mn, mx])

        return fig, axis

    def __str__(self):
        return f"... to be implemented ..."

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(true_pred_scatter_plot=plot_to_html_img(self.plot_true_pred_scatter))