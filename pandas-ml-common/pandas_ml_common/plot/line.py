from pandas_ml_common.utils import get_pandas_object
from pandas_ml_common.plot.utils import new_fig_ts_axis


def plot_line(df, fields, figsize=None, ax=None, **kwargs):
    data = get_pandas_object(df, fields)

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    ax.plot(df.index, data, **kwargs)
    return ax

