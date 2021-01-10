from pandas_ml_common.plot.utils import new_fig_ts_axis

from pandas_ml_common.utils import get_pandas_object
from pandas_ml_common.utils import has_indexed_columns


def plot_line(df, fields, figsize=None, ax=None, **kwargs):
    data = get_pandas_object(df, fields)

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    if has_indexed_columns(data):
        for col in data.columns:
            plot_line(data, col, figsize, ax, **kwargs)
    else:
        ax.plot(df.index, data.values, label=str(data.name), **kwargs)

    return ax

