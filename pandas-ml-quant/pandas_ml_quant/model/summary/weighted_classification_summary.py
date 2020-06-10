import logging
import os

import numpy as np
from matplotlib.figure import Figure

from pandas_ml_common import Typing
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import ClassificationSummary

_log = logging.getLogger(__name__)


class WeightedClassificationSummary(ClassificationSummary):

    def __init__(self, df: Typing.PatchedDataFrame, **kwargs):
        super().__init__(df)

        # TODO we can calculate the gross loss from the predicted band and the true price,
        #  therefore we need to pass the true price as gross loss such that we calculate the real loss

    def plot_gross_confusion_matrix(self):
        # FIXME plot confusion matrix using gross amounts
        pass

    def plot_classification(self):
        # FIXME plot gross loss bubble chart
        return ""

    def _repr_html_(self):
        from mako.template import Template

        path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(path, 'html', 'weighted_classification_summary.html')
        template = Template(filename=file)

        return template.render(
            summary=self,
            gross_confusion=self.gross_confusion(),
            roc_plot=plot_to_html_img(self.plot_ROC),
            cmx_plot=plot_to_html_img(self.plot_confusion_matrix),
            gross_loss_plot=plot_to_html_img(self.plot_classification),
        )
