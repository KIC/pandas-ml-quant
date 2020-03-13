from pandas_ml_common import get_pandas_object
from pandas_ml_quant.plots.utils import new_fig_ts_axis


def ta_line(df, fields, figsize=None, ax=None, **kwargs):
    data = get_pandas_object(df, fields).values

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    ax.plot(df.index, data, **kwargs)
    return ax

