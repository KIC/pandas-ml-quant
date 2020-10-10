from collections import namedtuple
from typing import List

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils.numpy_utils import clean_one_hot_classification
from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from pandas_ml_utils import html
from pandas_ml_utils.constants import *
from .figures import *


class Summary(object):
    """
    Summary objects a used to visually present the results of a `df.fit` fitted model or a `df.backtest`
    All implementations of `Summary` need to:
     * pass a `pd.DataFrame` to `super().__init()__`
     * implement `_repr_html_()`
    """

    Cell = namedtuple("Cell", ["index", "rowspan", "colspan"])

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', *args, layout: List[List[int]] = None, **kwargs):
        self._df = df
        self.model = model
        self.args = args
        self.kwargs = kwargs

        if layout is not None:
            grid = np.array(layout, dtype=int)
            table: List[List[Summary.Cell]] = [[]]
            for i, j in np.ndindex(grid.shape):
                if i > 0:
                    if grid[i-1, j] == grid[i, j]:
                        # rowspan += 1
                        cell = table[-1][j]
                        table[-1][j] = Summary.Cell(cell.index, cell.rowspan + 1, cell.colspan)

                    if j <= 0:
                        # we need to add a new empty row
                        table.append([])
                if j > 0:
                    if grid[i, j-1] == grid[i, j]:
                        # colspan += 1
                        cell = table[i][j-1]
                        table[i][-1] = Summary.Cell(cell.index, cell.rowspan, cell.colspan + 1)
                    else:
                        table[i].append(Summary.Cell(grid[i, j], 1, 1))
                else:
                    table[i].append(Summary.Cell(grid[i, j], 1, 1))

            # assign layout
            if len(table) > 0 and table[-1] == []:
                table.pop()

            self.layout = table
        else:
            self.layout = None

    @property
    def df(self):
        return self._df

    def _repr_html_(self):
        from mako.template import Template
        from mako.lookup import TemplateLookup
        plot = "<class 'matplotlib.figure.Figure'>"

        figures = [arg(self.df, model=self.model) for arg in self.args]
        figures = [("img", plot_to_html_img(f)) if str(type(f)) == plot else ("html", f._repr_html_()) for f in figures]

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(summary=self, figures=figures, layout=self.layout)


class RegressionSummary(Summary):

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', **kwargs):
        super().__init__(
            df,
            model,
            plot_true_pred_scatter,
            df_tail,
            layout=[[0, -1], [1, 1]],
            **kwargs
        )
        # TODO we should also add some figures as table like r2 ...

    def __str__(self):
        return f"... to be implemented ... "  # return r2 and such


class ClassificationSummary(Summary):

    def __init__(self, df: Typing.PatchedDataFrame, model: 'Model', **kwargs):
        super().__init__(
            df,
            model,
            plot_confusion_matrix,
            plot_receiver_operating_characteristic,
            df_tail,
            layout=[[0, 1], [2, 2]],
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

