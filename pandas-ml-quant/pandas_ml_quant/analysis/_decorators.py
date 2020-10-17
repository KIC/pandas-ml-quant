from typing import Union

import pandas as pd

from pandas_ml_common import has_indexed_columns


def for_each_column(func):
    def exec_on_each_column(df: Union[pd.DataFrame, pd.Series], *args, **kwargs):
        def exec_on_group(group):
            res = func(df[group], *args, **kwargs)
            if isinstance(res, pd.DataFrame) and len(res.columns) > 1:
                return res.add_multi_index(group, inplace=True)
            else:
                return res

        if has_indexed_columns(df):
            groups = [exec_on_group(group) for group in df.columns.to_list()]
            return pd.concat(groups, axis=1)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_column


def for_each_top_level_column(func):
    def exec_on_each_tl_column(df: pd.DataFrame, *args, **kwargs):
        if isinstance(df.columns, pd.MultiIndex):
            groups = [func(df[group], *args, **kwargs).to_frame().add_multi_index(group, inplace=True) for group in df.unique_level_columns(0)]
            return pd.concat(groups, axis=1)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_tl_column


def for_each_top_level_row(func):
    def exec_on_each_tl_row(df: pd.DataFrame, *args, **kwargs):
        if isinstance(df.index, pd.MultiIndex):
            return pd.concat(
                [func(df.loc[group], *args, **kwargs).add_multi_index(group, inplace=True, axis=0) for group in df.unique_level_rows(0)],
                axis=0
            )
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_tl_row
