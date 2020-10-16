from typing import Union

import pandas as pd
import numpy as np

from pandas_ml_common import has_indexed_columns


def sort_distance(s1: pd.Series, s2: pd.Series, top_percent=None) -> np.ndarray:
    s = (s1 / s2 - 1).dropna().abs()
    distances = s.sort_values().values

    if top_percent is not None:
        return distances[:int(np.ceil(len(distances) * top_percent))]
    else:
        return distances


def symmetric_quantile_factors(a, bins=11):
    return np.linspace(1-a, 1+a, bins)


def conditional_func(s: pd.Series, s_true: pd.Series, s_false: pd.Series):
    df = s.to_frame("CONDITION").join(s_true.rename("TRUTHY")).join(s_false.rename("FALSY"))
    return df.apply(lambda r: r["TRUTHY"] if r["CONDITION"] is True else r["FALSY"] , axis=1)


def difference(a: pd.Series, b: pd.Series, relative: bool, replace_inf=0):
    if relative:
        if replace_inf is not None:
            return (a / b - 1).replace([np.inf, -np.inf], replace_inf)
        else:
            return a / b - 1
    else:
        return a - b


def for_each_column(func):
    def exec_on_each_column(df: Union[pd.DataFrame, pd.Series], *args, **kwargs):
        if has_indexed_columns(df):
            groups = [func(df[group], *args, **kwargs).add_multi_index(group) for group in df.columns.to_list()]
            return pd.concat(groups, axis=1)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_column


def for_each_top_level_column(func):
    def exec_on_each_column(df: pd.DataFrame, *args, **kwargs):
        if isinstance(df.columns, pd.MultiIndex):
            groups = [func(df[group], *args, **kwargs).to_frame().add_multi_index(group) for group in df.unique_level_columns(0)]
            return pd.concat(groups, axis=1)
        else:
            return func(df, *args, **kwargs)

    return exec_on_each_column