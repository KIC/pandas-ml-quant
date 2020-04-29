import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from sklearn.metrics import f1_score

from pandas_ml_common import Typing
from pandas_ml_common.serialization_utils import plot_to_html_img
from pandas_ml_common.utils import unique_level_columns
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import Summary

_log = logging.getLogger(__name__)


# FIXME this need to be a specialized summary
class BinaryWeightedClassificationSummary(Summary):

    def __init__(self, df: Typing.PatchedDataFrame, **kwargs):
        super().__init__(df)
        self.probability_cutoff = 0.5
        self.confusions = BinaryWeightedClassificationSummary._calculate_confusions(df, self.probability_cutoff)

    def set_probability_cutoff(self, probability_cutoff: float):
        self.probability_cutoff = probability_cutoff
        self.confusions = BinaryWeightedClassificationSummary._calculate_confusions(self.df, probability_cutoff)

    def get_confusion_matrix(self, total=True):
        confusion_mx = np.array([[[len(c) for c in r] for r in cm] for cm in self.confusions])
        return confusion_mx.sum(axis=0) if total else confusion_mx

    def get_confusion_loss(self, total=True):
        try:
            loss_mx = np.array([[[c[GROSS_LOSS_COLUMN_NAME].iloc[:, 0].sum() for c in r] for r in cm] for cm in self.confusions])
            return loss_mx.sum(axis=0) if total else loss_mx
        except (ValueError, ArithmeticError):
            return np.array([[0, 0], [0, 0]])

    def get_metrics(self):
        fp_ratio, fn_ratio = self.get_ratios()
        df = self.df
        pc = self.probability_cutoff

        if df.columns.nlevels == 3:
            f1 = np.array([f1_score(df[target, LABEL_COLUMN_NAME].iloc[:, 0].values,
                                    df[target, PREDICTION_COLUMN_NAME].iloc[:, 0].values > pc)
                           for target in unique_level_columns(df)]).mean()
        else:
            f1 = f1_score(df[LABEL_COLUMN_NAME].iloc[:, 0].values,
                          df[PREDICTION_COLUMN_NAME].iloc[:, 0].values > pc)

        return {"FP Ratio": fp_ratio,
                "FN Ratio": fn_ratio,
                "F1 Score": f1}

    def get_ratios(self):
        cm = self.get_confusion_matrix()
        return cm[0, 1] / cm[0, 0], cm[1, 0] / cm[0, 0]

    def plot_classification(self, figsize=(16, 9)) -> Figure:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from pandas.plotting import register_matplotlib_converters

        # get rid of deprecation warning
        register_matplotlib_converters()
        pc = self.probability_cutoff
        df = self.df

        # define grid
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

        x = np.arange(len(df))
        yp = df[PREDICTION_COLUMN_NAME].iloc[:, 0]
        yl = df[GROSS_LOSS_COLUMN_NAME].iloc[:, 0]

        pos = ((df[PREDICTION_COLUMN_NAME].iloc[:, 0] <  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] < pc)) |\
              ((df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] > pc))

        neg = ((df[PREDICTION_COLUMN_NAME].iloc[:, 0] >= pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] < pc)) |\
              ((df[PREDICTION_COLUMN_NAME].iloc[:, 0] <  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] > pc))

        # plot true positives/negatives
        sns.scatterplot(ax=ax1,
                        x=x[pos],
                        y=yl[pos],
                        size=np.abs(yp[pos].values - 0.5),
                        size_norm=(0, 1),
                        color=sns.xkcd_rgb['pale green'],
        )

        # plot false positives/negatives
        sns.scatterplot(ax=ax1,
                        x=x[neg],
                        y=yl[neg],
                        size=np.abs(yp[neg].values - 0.5),
                        size_norm=(0, 0.5),
                        color=sns.xkcd_rgb['cerise'],
        )

        # make y symmetric
        maax = np.abs(yl.dropna().values).max()
        ax1.set_ylim((-maax, maax))

        # add mean line
        ax1.hlines(0, 0, len(df), color=sns.xkcd_rgb['silver'])

        return fig

    @staticmethod
    def _calculate_confusions(df, probability_cutoff):
        if df.columns.nlevels == 3:
            # multiple targets
            return [BinaryWeightedClassificationSummary._calculate_confusions(df[target], probability_cutoff)[0]
                    for target in unique_level_columns(df)]
        else:
            pc = probability_cutoff
            tp = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] >  pc)]
            fp = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] <= pc)]
            tn = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] <= pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] <= pc)]
            fn = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] <= pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] >  pc)]

            return [[[tp, fp],
                     [fn, tn]]]

    def _repr_html_(self):
        from mako.template import Template

        file = os.path.abspath(__file__)
        path = os.path.join(os.path.dirname(file), 'html')
        file = os.path.basename(file).replace('.py', '.html')
        template = Template(filename=os.path.join(path, file))

        return template.render(
            summary=self,
            gross_loss_plot=plot_to_html_img(self.plot_classification),
        )
