import logging
import os
from typing import Dict

import numpy as np
from sklearn.metrics import f1_score

from pandas_ml_common import pd
from pandas_ml_common.utils import unique_level_columns
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import Summary

_log = logging.getLogger(__name__)

# FIXME this need to be a specialized summary and moved over to the qunat library!!
class BinaryClassificationSummary(Summary):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)
        self.probability_cutoff = 0.5
        self.confusions = BinaryClassificationSummary._calculate_confusions(df, self.probability_cutoff)

    def set_probability_cutoff(self, probability_cutoff: float):
        self.probability_cutoff = probability_cutoff
        self.confusions = BinaryClassificationSummary._calculate_confusions(self.df, probability_cutoff)

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

    def plot_classification(self, figsize=(16, 9)) -> Dict:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from pandas.plotting import register_matplotlib_converters

        # get rid of deprecation warning
        register_matplotlib_converters()

        probability_cutoff = self.probability_cutoff
        pc = self.probability_cutoff
        plots = {}

        for target in unique_level_columns(self.df) if self.df.columns.nlevels == 3 else [None]:
            # get target and frame
            df = self.df[target] if target is not None else self.df

            # define grid
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])

            # plot probability
            bar = sns.lineplot(x=range(len(df)), y=df[PREDICTION_COLUMN_NAME].iloc[:, 0], ax=ax0)
            ax0.hlines(probability_cutoff, 0, len(df), color=sns.xkcd_rgb['silver'])

            # plot loss
            color = pd.Series(0, index=df.index)
            color.loc[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & df[LABEL_COLUMN_NAME].iloc[:, 0] > pc] = 1
            color.loc[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] <= pc) & df[LABEL_COLUMN_NAME].iloc[:, 0] > pc] = 2

            colors = {0: sns.xkcd_rgb['white'], 1: sns.xkcd_rgb['pale green'], 2: sns.xkcd_rgb['cerise']}
            palette = [colors[color_index] for color_index in np.sort(color.unique())]

            sns.scatterplot(ax=ax1,
                            x=range(len(df)),
                            y=df[GROSS_LOSS_COLUMN_NAME].iloc[:, 0].clip(upper=0),
                            size=df[GROSS_LOSS_COLUMN_NAME].iloc[:, 0] * -1,
                            hue=color,
                            palette=palette)

            plt.close()
            plots[target] = fig

        return plots

    @staticmethod
    def _calculate_confusions(df, probability_cutoff):
        if df.columns.nlevels == 3:
            # multiple targets
            return [BinaryClassificationSummary._calculate_confusions(df[target], probability_cutoff)[0]
                    for target in unique_level_columns(df)]
        else:
            pc = probability_cutoff
            tp = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] >  pc)]
            fp = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] <= pc)]
            tn = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] <= pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] <= pc)]
            fn = df[(df[PREDICTION_COLUMN_NAME].iloc[:, 0] <= pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] >  pc)]

            return [[[tp, fp],
                     [fn, tn]]]

    def _html_template_file(self):
        return f"{os.path.abspath(__file__)}.html"
