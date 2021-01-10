import numpy as np
from pandas_ml_common.plot.utils import new_fig_ts_axis

from pandas_ml_common.utils import get_pandas_object


def plot_matrix(df, fields, figsize=None, ax=None, **kwargs):
    data = fields if isinstance(fields, np.ndarray) else (get_pandas_object(df, fields)._.values.squeeze())

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    ax.matshow(data)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return ax
