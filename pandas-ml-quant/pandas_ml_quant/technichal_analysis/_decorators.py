from functools import wraps
from typing import Union

import pandas as pd

from pandas_ml_common import has_indexed_columns, unique_level_rows, unique_level_columns


def for_each_column(func):
    @wraps(func)
    def exec_on_each_column(df: pd.DataFrame, *args, **kwargs):
        if df.ndim > 1 and df.shape[1] > 0:
            results = [func(df[col], *args, **kwargs) for col in df.columns]
            if results[0].ndim > 1 and results[0].shape[1] > 1:
                for i, col in enumerate(df.columns):
                    results[i].columns = pd.MultiIndex.from_product([[col], results[i].columns.tolist()])

            return pd.concat(results, axis=1, join='inner')
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_column


def for_each_top_level_column(func):
    @wraps(func)
    def exec_on_each_tl_column(df: pd.DataFrame, *args, **kwargs):
        if df.ndim > 1 and isinstance(df.columns, pd.MultiIndex):
            groups = [func(df[group], *args, **kwargs).to_frame().add_multi_index(group, inplace=True) for group in unique_level_columns(df, 0)]
            return pd.concat(groups, axis=1)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_tl_column


def for_each_top_level_row(func):
    @wraps(func)
    def exec_on_each_tl_row(df: pd.DataFrame, *args, **kwargs):
        if isinstance(df.index, pd.MultiIndex) and df.ndim > 1:
            top_level = unique_level_rows(df, 0)
            if len(top_level) > 1:
                return pd.concat(
                    [func(df.loc[group], *args, **kwargs).add_multi_index(group, inplace=True, axis=0) for group in top_level],
                    axis=0
                )
            else:
                return func(df, *args, **kwargs)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_tl_row
