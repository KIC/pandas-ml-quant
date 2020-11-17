import logging
import os

import numpy as np
from matplotlib.figure import Figure

from pandas_ml_common import Typing
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import Model
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import ClassificationSummary

_log = logging.getLogger(__name__)


class BinaryWeightedClassificationSummary(ClassificationSummary):

    def __init__(self, df: Typing.PatchedDataFrame, model: Model, probability_cutoff=0.5, **kwargs):
        super().__init__(df, model, **kwargs)
        pc = self.probability_cutoff = probability_cutoff

        self.true_neg = (df[PREDICTION_COLUMN_NAME].iloc[:, 0] <  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] < pc)
        self.true_pos = (df[PREDICTION_COLUMN_NAME].iloc[:, 0] >  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] > pc)
        self.pos_idx = self.true_pos | self.true_neg

        self.false_pos = (df[PREDICTION_COLUMN_NAME].iloc[:, 0] >= pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] < pc)
        self.false_neg = (df[PREDICTION_COLUMN_NAME].iloc[:, 0] <  pc) & (df[LABEL_COLUMN_NAME].iloc[:, 0] > pc)
        self.neg_idx = self.false_pos | self.false_neg

    def set_probability_cutoff(self, probability_cutoff: float):
        return BinaryWeightedClassificationSummary(self.df, probability_cutoff, **self.kwargs)

    def gross_confusion(self):
        return (
            len(self.df),
            self.df[GROSS_LOSS_COLUMN_NAME][self.pos_idx].count().values.sum(),
            self.df[GROSS_LOSS_COLUMN_NAME][self.neg_idx].dropna().abs().values.sum()
        )

    def plot_gross_distribution(self, figsize=(16, 9)) -> Figure:
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

        # plot true positives/negatives
        sns.scatterplot(ax=ax1,
                        x=x[self.pos_idx],
                        y=yl[self.pos_idx],
                        size=np.abs(yp[self.pos_idx].values - 0.5),
                        size_norm=(0, 0.5),
                        color=sns.xkcd_rgb['pale green'],
        )

        # plot false positives/negatives
        sns.scatterplot(ax=ax1,
                        x=x[self.neg_idx],
                        y=yl[self.neg_idx],
                        size=np.abs(yp[self.neg_idx].values - 0.5),
                        size_norm=(0, 0.5),
                        color=sns.xkcd_rgb['cerise'],
        )

        # make y symmetric
        maax = np.abs(yl.dropna().values).max()
        ax1.set_ylim((-maax, maax))

        # add mean line
        ax1.hlines(0, 0, len(df), color=sns.xkcd_rgb['silver'])

        return fig

    def _repr_html_(self):
        from mako.template import Template

        path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(path, 'html', 'binary_weighted_classification_summary.html')
        template = Template(filename=file)

        return template.render(
            summary=self,
            gross_confusion=self.gross_confusion(),
            roc_plot=plot_to_html_img(self.plot_ROC),
            cmx_plot=plot_to_html_img(self.plot_confusion_matrix),
            gross_distribution_plot=plot_to_html_img(self.plot_gross_distribution),
        )


# TODO create a generic MultiModelMultiSummary(summary_provider, ...)
class MultipleBinaryWeightedClassificationSummary(ClassificationSummary):

    def __init__(self, df: Typing.PatchedDataFrame, probability_cutoff=0.5, **kwargs):
        super().__init__(df)
        if GROSS_LOSS_COLUMN_NAME in df:
            df = df[[LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME, GROSS_LOSS_COLUMN_NAME]]
        else:
            _log.warning("No gross loss provided!")
            df = df[[LABEL_COLUMN_NAME, PREDICTION_COLUMN_NAME]]

        nr_columns = len(df.columns) // 3

        self.binary_summaries =\
            [BinaryWeightedClassificationSummary(df.iloc[:,[i, i+nr_columns, i+nr_columns*2]], probability_cutoff, **kwargs)
             for i in range(nr_columns)]

    def set_probability_cutoff(self, probability_cutoff: float):
        return MultipleBinaryWeightedClassificationSummary(self.df, probability_cutoff, **self.kwargs)

    def _repr_html_(self):
        from mako.template import Template

        path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(path, 'html', 'multi_binary_weighted_classification_summary.html')
        template = Template(filename=file)

        return template.render(
            summary=self,
            gross_confusions=[bs.gross_confusion() for bs in self.binary_summaries],
            roc_plots=[plot_to_html_img(bs.plot_ROC) for bs in self.binary_summaries],
            cmx_plots=[plot_to_html_img(bs.plot_confusion_matrix) for bs in self.binary_summaries],
            # gross_loss_plot=[plot_to_html_img(bs.plot_classification) for bs in self.binary_summaries],
        )

