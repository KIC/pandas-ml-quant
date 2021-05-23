from functools import wraps

import pandas as pd
import logging
from pandas_ml_common import unique_level_rows, unique_level_columns
from pandas_ml_common.utils import same_columns_after_level

_log = logging.getLogger(__name__)


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


def for_each_top_level_column(func, level=0):
    @wraps(func)
    def exec_on_each_tl_column(df: pd.DataFrame, *args, **kwargs):
        if df.ndim > 1 and isinstance(df.columns, pd.MultiIndex):
            # check if the shape of the 2nd level is identical else threat as if not multi index
            if same_columns_after_level(df, level):
                top_level = unique_level_columns(df, level)
                groups = [func(df.xs(group, axis=1, level=level), *args, **kwargs).to_frame().add_multi_index(group, inplace=True, level=level) for group in top_level]
                return pd.concat(groups, axis=1)
            else:
                _log.warning(f"columns in further levels do not follow the same structure! Treat as normal Index")
                return func(df, *args, **kwargs)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_tl_column


def for_each_top_level_row(func):
    @wraps(func)
    def exec_on_each_tl_row(df: pd.DataFrame, *args, **kwargs):
        if isinstance(df.index, pd.MultiIndex):
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


def is_time_consuming(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper._is_timeconsuming = True
    return wrapper
