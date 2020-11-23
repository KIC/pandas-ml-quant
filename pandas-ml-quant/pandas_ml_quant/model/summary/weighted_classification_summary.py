import logging
import os

import numpy as np

from pandas_ml_common import Typing, pd
from pandas_ml_common.utils.numpy_utils import empty_lists, get_buckets, clean_one_hot_classification
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import Model
from pandas_ml_utils.constants import *
from pandas_ml_utils.ml.summary import ClassificationSummary

_log = logging.getLogger(__name__)


class WeightedClassificationSummary(ClassificationSummary):

    def __init__(self, df: Typing.PatchedDataFrame, model: Model, clip_profit_at=0, classes=None, **kwargs):
        super().__init__(df, model, **kwargs)
        self.clip_profit_at = clip_profit_at
        self.targets = df[TARGET_COLUMN_NAME]

        # calculate confusion indices
        tv, pv = clean_one_hot_classification(self.df[LABEL_COLUMN_NAME]._.values, self.df[PREDICTION_COLUMN_NAME]._.values)

        # confusion matrix needs integer encoding
        tv = np.apply_along_axis(np.argmax, 1, tv)
        pv = np.apply_along_axis(np.argmax, 1, pv)
        distinct_values = len({*tv.flatten()}) if classes is None else classes
        cm = empty_lists((distinct_values, distinct_values))

        for i, (t, p) in enumerate(zip(tv, pv)):
            cm[int(t), int(p)].append(self.df.index[i])

        self.confusion_indices = cm

        # we can calculate the gross loss from the predicted band and the true price,
        #  therefore we need to pass the true price as gross loss such that we calculate the real loss
        self.df_gross_loss = pd.DataFrame({
            "bucket": df[[TARGET_COLUMN_NAME]].apply(get_buckets, axis=1, raw=True),
            "pidx": df.apply(lambda r: int(r[PREDICTION_COLUMN_NAME]._.values.argmax()), axis=1, raw=False),
            "price": df[GROSS_LOSS_COLUMN_NAME].values[:,0]
        }, index=df.index)

        # find target for predicted value
        mid = self.targets.shape[1] / 2.0
        self.df_gross_loss["loss"] = self.df_gross_loss.apply(
            lambda r: (r["price"] - r["bucket"][r["pidx"]][0]) if r["pidx"] <= mid else (
                        r["bucket"][r["pidx"]][1] - r["price"]),
            axis=1,
            raw=False
        ).fillna(0)


    def plot_gross_confusion(self, figsize=(9, 8)):
        import matplotlib.pyplot as plt
        import seaborn as sns

        f, ax = plt.subplots(figsize=figsize)
        sns.heatmap(self._gross_confusion(), annot=True, linewidths=.5, ax=ax)
        return f, ax

    def _gross_confusion(self):
        cm = np.empty(self.confusion_indices.shape)
        for i in np.ndindex(cm.shape):
            cm[i] = self.df_gross_loss["loss"].loc[self.confusion_indices[i]].clip(upper=self.clip_profit_at).mean()

        return pd.DataFrame(cm)

    def plot_samples(self, worst=200):
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(1, 1, figsize=(16, 9))

        _df = self.df
        _df = _df.loc[self.df_gross_loss["loss"].sort_values().index[:worst]]

        ax.imshow(
            _df[PREDICTION_COLUMN_NAME]._.values.T,
            cmap='afmhot',
            interpolation='nearest',
            aspect='auto'
        )

        ax.scatter(
            np.arange(0, len(_df)),
            _df[LABEL_COLUMN_NAME].apply(lambda r: np.array(r[0]).argmax(), axis=1, raw=True)
        )

        ax.invert_yaxis()
        return f, ax

    def _repr_html_(self):
        from mako.template import Template

        path = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(path, 'html', 'weighted_classification_summary.html')
        template = Template(filename=file)

        # TODO add min premium field -> sum all losses / nr_of_trades
        return template.render(
            summary=self,
            gmx_plot=plot_to_html_img(self.plot_gross_confusion),
            roc_plot=plot_to_html_img(self.plot_ROC),
            cmx_plot=plot_to_html_img(self.plot_confusion_matrix),
            samples_plot=plot_to_html_img(self.plot_samples)
        )
