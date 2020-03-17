from typing import Tuple

import numpy as np
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import roc_curve, auc, confusion_matrix as sk_confusion_matrix

from pandas_ml_common import pd
from pandas_ml_common.serialization_utils import plot_to_html_img
from pandas_ml_utils import html
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import Summary


class ClassificationSummary(Summary):

    def __init__(self, df: pd.DataFrame, true_columns=None, pred_columns=None):
        super().__init__(df)
        self.true_columns = LABEL_COLUMN_NAME if true_columns is None else true_columns
        self.pred_columns = PREDICTION_COLUMN_NAME if pred_columns is None else pred_columns

    def plot_ROC(self, figsize=(6, 6)) -> Tuple[Figure, Axis]:
        import matplotlib.pyplot as plt

        # get true and prediction data
        truth = self.df[self.true_columns].ml.values.reshape((len(self.df), -1))
        prediction = self.df[self.pred_columns].ml.values.reshape(truth.shape)

        # calculate legends
        legend = [(col[1] if isinstance(col, tuple) else col) for col in self.df[self.true_columns].columns.tolist()]
        if truth.shape[1] > len(legend):
            legend = list(range(truth.shape[1]))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        if len(truth.shape) > 1:
            for i in range(truth.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(truth[:, i], prediction[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[0], tpr[0], _ = roc_curve(truth, prediction)
            roc_auc[0] = auc(fpr[0], tpr[0])

        # plot ROC curves
        fig, axis = plt.subplots(figsize=figsize)

        for i in fpr.keys():
            plt.plot(fpr[i], tpr[i], label=f"{legend[i]} auc:{roc_auc[i] * 100:.2f}")

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        return fig, axis

    def plot_confusion_matrix(self, figsize=(6, 6)) -> Tuple[Figure, Axis]:
        from mlxtend.plotting import plot_confusion_matrix

        truth, prediction = self._fix_label_prediction_representation()

        distinct_values = {*truth.reshape((-1,))}
        cm = confusion_matrix(truth, prediction, binary=len(distinct_values) <= 2)
        return plot_confusion_matrix(cm, figsize=figsize)

    def _fix_label_prediction_representation(self):
        true_values = self.df[self.true_columns].ml.values
        if len(true_values.shape) > 1:
            true_values = true_values.reshape((len(true_values), -1))

        pred_values = self.df[self.pred_columns].ml.values.reshape(true_values.shape)

        if len(true_values.shape) > 1 and true_values.shape[1] > 1:
            true_values = np.apply_along_axis(np.argmax, 1, true_values)
            pred_values = np.apply_along_axis(np.argmax, 1, pred_values)

        return true_values.reshape((-1, 1)), pred_values.reshape((-1, 1))

    def __str__(self):
        cmx = sk_confusion_matrix(*self._fix_label_prediction_representation())
        return f"{cmx}"

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(fit=self,
                               roc_plot=plot_to_html_img(self.plot_ROC),
                               cmx_plot=plot_to_html_img(self.plot_confusion_matrix))