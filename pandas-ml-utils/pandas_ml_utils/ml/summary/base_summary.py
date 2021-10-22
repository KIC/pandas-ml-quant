from collections import namedtuple
from typing import List, Union, Any, Callable

import pandas as pd

from pandas_ml_common.utils.serialization_utils import plot_to_html_img
from .figures import *


class Summary(object):
    """
    Summary objects a used to visually present the results of a `df.fit` fitted model or a `df.backtest`
    All implementations of `Summary` need to:
     * pass a `pd.DataFrame` to `super().__init()__`
     * implement `_repr_html_()`
    """

    Cell = namedtuple("Cell", ["index", "rowspan", "colspan"])

    @staticmethod
    def provide(summary_provider: Callable[..., 'Summary'], prediction_df: MlTypes.PatchedDataFrame, model: 'Model', source_df: MlTypes.PatchedDataFrame, **kwargs):
        return call_callable_dynamic_args(summary_provider, df=prediction_df, model=model, source=source_df, **kwargs)

    def __init__(self, df: MlTypes.PatchedDataFrame, *args: Callable[..., Union[MlTypes.PatchedDataFrame, Any]], layout: List[List[int]] = None, **kwargs):
        self._df = df
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
                        cell = table[i][-1]
                        table[i][-1] = Summary.Cell(cell.index, cell.rowspan, cell.colspan + 1)
                    else:
                        table[i].append(Summary.Cell(grid[i, j], 1, 1))
                else:
                    table[i].append(Summary.Cell(grid[i, j], 1, 1))

            # assign layout
            if len(table) > 0 and table[-1] == []:
                table.pop()

            self.layout = table
            self.laypout_nr_columns = grid.shape[1]
        else:
            self.layout = None
            self.laypout_nr_columns = None

    @property
    def df(self):
        return self._df

    def __str__(self):
        return str(self.df.groupby(level=0).tail(1)) if isinstance(self.df.index, pd.MultiIndex) else str(self.df.tail())

    def _repr_html_(self):
        from pandas_ml_utils import html
        from mako.template import Template
        from mako.lookup import TemplateLookup
        plot = "<class 'matplotlib.figure.Figure'>"

        figures = [call_callable_dynamic_args(arg, df=self.df, **self.kwargs) for arg in self.args]
        figures = [("img", plot_to_html_img(f)) if str(type(f)) == plot else ("html", f._repr_html_() if hasattr(f, "_repr_html_") else str(f)) for f in figures]

        template = Template(filename=html.SELF_TEMPLATE(__file__), lookup=TemplateLookup(directories=['/']))
        return template.render(summary=self, figures=figures, layout=self.layout, laypout_nr_columns=self.laypout_nr_columns)


class RegressionSummary(Summary):

    @staticmethod
    def factory(include_feature_importance=True):
        return lambda df, model, source, **kwargs: RegressionSummary(df, model, source, include_feature_importance, **kwargs)

    def __init__(self, df: MlTypes.PatchedDataFrame, model: 'Model', source: MlTypes.PatchedDataFrame, include_feature_importance: bool = True, **kwargs):
        super().__init__(
            df,
            plot_true_pred_scatter,                                             # 0
            df_regression_scores,                                               # 1
            plot_feature_importance if include_feature_importance else None,    # 2
            df_tail,                                                            # 3
            layout=[[0, 1],
                    [2, 2],
                    [3, 3]] if include_feature_importance else [[0, 1],
                                                                [3, 3]],
            model=model,
            source=source,
            **kwargs
        )


class ClassificationSummary(Summary):

    @staticmethod
    def factory(include_feature_importance=True):
        return lambda df, model, source, **kwargs: ClassificationSummary(df, model, source, include_feature_importance, **kwargs)

    def __init__(self, df: MlTypes.PatchedDataFrame, model: 'Model', source: pd.DataFrame, include_feature_importance: bool = True, **kwargs):
        super().__init__(
            df,
            plot_confusion_matrix,                                              # 0
            plot_receiver_operating_characteristic,                             # 1
            df_classification_scores,                                           # 2
            plot_feature_importance if include_feature_importance else None,    # 3
            df_tail,                                                            # 4
            layout=([[2, 0, 0, 1, 1],
                     [3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4]] if include_feature_importance else [[2, 0, 0, 1, 1],
                                                                          [4, 4, 4, 4, 4]]),
            model=model,
            source=source,
            **kwargs
        )

    def __str__(self):
        from mlxtend.evaluate import confusion_matrix

        # get true and prediction data. It needs to be a one hot encoded 2D array [samples, class] where nr_classes >= 2
        tv, pv = clean_one_hot_classification(self.df[LABEL_COLUMN_NAME].ML.values, self.df[PREDICTION_COLUMN_NAME].ML.values)

        # confusion matrix needs integer encoding
        tv = np.apply_along_axis(np.argmax, 1, tv)
        pv = np.apply_along_axis(np.argmax, 1, pv)
        cm = confusion_matrix(tv, pv, binary=tv.max() < 2)

        return f"{cm}"

