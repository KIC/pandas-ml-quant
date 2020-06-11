import logging
import os

import numpy as np
from matplotlib.figure import Figure

from pandas_ml_common import Typing, pd
from pandas_ml_common.utils.numpy_utils import empty_lists, get_buckets
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import ClassificationSummary

_log = logging.getLogger(__name__)


class WeightedClassificationSummary(ClassificationSummary):

    def __init__(self, df: Typing.PatchedDataFrame, **kwargs):
        super().__init__(df)
        self.confusion_indices = self._confusion_indices()
        self.targets = df[TARGET_COLUMN_NAME]

        # we can calculate the gross loss from the predicted band and the true price,
        #  therefore we need to pass the true price as gross loss such that we calculate the real loss
        self.df_gross_loss = pd.DataFrame({
            "bucket": df[[TARGET_COLUMN_NAME]].apply(get_buckets, axis=1, raw=True),
            "pidx": df.apply(lambda r: int(r[PREDICTION_COLUMN_NAME]._.values.argmax()), axis=1, raw=False),
            "price": df[GROSS_LOSS_COLUMN_NAME].values[:,0]
        }, index=df.index)

        # find target for predicted value
        mid = self.targets.shape[1] / 2.0
        if self.targets.shape[1] % 2 == 0:
            # 0             1            2            3           4
            #   25.901625  	  26.815800 	27.729975 	28.644150
            self.df_gross_loss["loss"] = self.df_gross_loss.apply(
                lambda r: (r["price"] - r["bucket"][r["pidx"]][0]) if r["pidx"] <= mid else (
                            r["bucket"][r["pidx"]][1] - r["price"]),
                axis=1,
                raw=False
            )
        else:
            # mid -> np.floor(mid)[1] / np.ceil(mid)[0]
            # 0           1            2           3
            #   25.901625 	26.815800 	 27.729975
            self.df_gross_loss["loss"] = self.df_gross_loss.apply(
                lambda r: (r["price"] - r["bucket"][r["pidx"]][1]) if r["pidx"] < mid else (r["bucket"][r["pidx"]][0] - r["price"]),
                axis=1,
                raw=False
            )

    def _confusion_indices(self):
        truth, prediction = self._fix_label_prediction_representation()
        distinct_values = len({*truth.reshape((-1,))})
        cm = empty_lists((distinct_values, distinct_values))

        for i, (t, p) in enumerate(zip(truth, prediction)):
            cm[int(t), int(p)].append(self.df.index[i])

        return cm

    def plot_gross_confusion(self):
        # TODO we can calculate the gross loss from the predicted band and the true price,
        #  therefore we need to pass the true price as gross loss such that we calculate the real loss

        pass

    def plot_gross_distribution(self):
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
