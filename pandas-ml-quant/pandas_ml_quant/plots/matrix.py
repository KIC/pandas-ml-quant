from pandas_ml_common import get_pandas_object
from pandas_ml_quant.plots.utils import new_fig_ts_axis
import numpy as np


def ta_matrix(df, fields, figsize=None, ax=None, **kwargs):
    data = fields if isinstance(fields, np.ndarray) else (get_pandas_object(df, fields).ml.values.squeeze())

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    ax.matshow(data)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return ax
