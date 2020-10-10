import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils.numpy_utils import clean_one_hot_classification
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import html
from pandas_ml_utils.constants import *


class Summary(object):
    """
    Summary objects a used to visually present the results of a `df.fit` fitted model or a `df.backtest`
    All implementations of `Summary` need to:
     * pass a `pd.DataFrame` to `super().__init()__`
     * implement `_repr_html_()`
    """

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', *args, **kwargs):
        self._df = df
        self.model = model
        self.args = args
        self.kwargs = kwargs

    @property
    def df(self):
        return self._df

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup

        figures = [arg(self.df, model=self.model) for arg in self.args]
        embedded_plots = [plot_to_html_img(f) for f in figures if str(type(f)) == "<class 'matplotlib.figure.Figure'>"]
        tables = [f for f in figures if str(type(f)) != "<class 'matplotlib.figure.Figure'>"]

        # TODO we need some table layouting ...
        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(summary=self, plots=embedded_plots, tables=tables)


class RegressionSummary(Summary):
    from .figures import plot_true_pred_scatter

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', **kwargs):
        super().__init__(df, model, RegressionSummary.plot_true_pred_scatter, **kwargs)
        # TODO we should also add some figures as table like r2 ...

    def __str__(self):
        return f"... to be implemented ... "  # return r2 and such


class ClassificationSummary(Summary):
    from .figures import plot_receiver_operating_characteristic, plot_confusion_matrix

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', **kwargs):
        super().__init__(
            df,
            model,
            ClassificationSummary.plot_confusion_matrix,
            ClassificationSummary.plot_receiver_operating_characteristic,
            **kwargs
        )
        # TODO we should also add some figures as table like f1 ...

    def __str__(self):
        from mlxtend.evaluate import confusion_matrix

        # get true and prediction data. It needs to be a one hot encoded 2D array [samples, class] where nr_classes >= 2
        tv, pv = clean_one_hot_classification(self.df[LABEL_COLUMN_NAME]._.values, self.df[PREDICTION_COLUMN_NAME]._.values)

        # confusion matrix needs integer encoding
        tv = np.apply_along_axis(np.argmax, 1, tv)
        pv = np.apply_along_axis(np.argmax, 1, pv)
        cm = confusion_matrix(tv, pv, binary=tv.max() < 2)

        return f"{cm}"

