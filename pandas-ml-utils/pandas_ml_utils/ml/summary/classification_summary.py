from typing import Tuple

import numpy as np
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import roc_curve, auc

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

        df = self.df
        dft = df[self.true_columns]
        dfp = df[self.pred_columns]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(dfp.columns)):
            fpr[i], tpr[i], _ = roc_curve(dft.values[:, i], dfp.values[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # plot ROC curves
        fig, axis = plt.subplots(figsize=figsize)

        for i in fpr.keys():
            plt.plot(fpr[i], tpr[i], label=f"{dfp.columns[i]} auc:{roc_auc[i] * 100:.2f}")
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

        df = self.df
        dft = df[self.true_columns]
        dfp = df[self.pred_columns]

        # eventually decode one hot encoded values
        if len(dft.shape) > 1 and dft.shape[1] > 1:
            dft = dft.apply(lambda row: np.argmax(row), raw=True, axis=1)
        else:
            dft = dft.astype(int)

        if len(dfp.shape) > 1 and dfp.shape[1] > 1:
            dfp = dfp.apply(lambda row: np.argmax(row), raw=True, axis=1)

        distinct_values = {*dft.values.reshape((-1,))}
        cm = confusion_matrix(dft.values, dfp.values, binary=len(distinct_values) <= 2)
        return plot_confusion_matrix(cm, figsize=figsize)

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(fit=self,
                               roc_plot=plot_to_html_img(self.plot_ROC),
                               cmx_plot=plot_to_html_img(self.plot_confusion_matrix))