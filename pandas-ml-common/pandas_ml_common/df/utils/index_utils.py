from collections import OrderedDict

import pandas as pd


def unique_level_columns(df: pd.DataFrame, level=0):
    return unique(df.columns.get_level_values(level)) if isinstance(df.columns, pd.MultiIndex) else df.columns


def unique_level_rows(df: pd.DataFrame, level=0):
    return unique(df.index.get_level_values(level)) if isinstance(df.index, pd.MultiIndex) else df.index


def unique(items):
    return list(OrderedDict.fromkeys(items))


def multi_index_shape(index: pd.MultiIndex):
    sets = [set() for _ in range(index.nlevels)]
    for tple in index.tolist():
        for i in range(index.nlevels):
            sets[i].add(tple[i])

    return tuple(len(x) for x in sets)
