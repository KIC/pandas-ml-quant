from typing import List

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from pandas_ml_common import get_pandas_object
from pandas_ml_quant.plots.utils import new_fig_ts_axis


def ta_bar(df, fields, figsize=None, ax=None, **kwargs):
    data = get_pandas_object(df, fields).values

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    ax.bar(df.index, height=data, **kwargs)
    return ax


def ta_stacked_bar(df, columns, figsize=None, ax=None, padding=0.02, **kwargs):
    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    if padding is not None:
        b, t = ax.get_ylim()

        if b == 0 and t == 1:
            b = np.inf
            t = -np.inf

        ax.set_ylim(min(df[columns].values.min(), b) * (1 - padding), max(df[columns].values.max(), t) * (1 + padding))

    bottom = None
    for column in columns:
        data = get_pandas_object(df, column)

        if bottom is not None:
            kwargs["bottom"] = bottom
            height = data - bottom
        else:
            height = data

        bottom = height if bottom is None else bottom + height
        ax.bar(mdates.date2num(df.index), height, **kwargs)

    return ax


def ta_bars(df, columns, figsize=None, ax=None, padding=0.02, **kwargs):
    # FIXME allow to pass pandas objects as columns
    columns = columns if isinstance(columns, List) else list(columns)

    if ax is None:
        fig, ax = new_fig_ts_axis(figsize)

    if padding is not None:
        b, t = ax.get_ylim()

        if b == 0 and t == 1:
            b = np.inf
            t = -np.inf

        ax.set_ylim(min(df[columns].values.min(), b) * (1 - padding), max(df[columns].values.max(), t) * (1 + padding))

    width = 1 * (1-padding) / len(columns)
    for i, column in enumerate(columns):
        ax.bar(mdates.date2num(df.index) + width * i, df[column].values, width, **kwargs)

    return ax
