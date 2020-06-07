import logging
import re
from collections import OrderedDict, Callable
from typing import List

import pandas as pd
import numpy as np
from pandas.core.base import PandasObject

from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args
from pandas_ml_common.utils.types import Constant

_log = logging.getLogger(__name__)


def has_indexed_columns(po: PandasObject):
    return hasattr(po, "columns") and isinstance(po.columns, pd.Index)


def add_multi_index(df, head, inplace=False):
    df = df if inplace else df.copy()
    df.columns = pd.MultiIndex.from_product([[head], df.columns.tolist()])
    return df


def inner_join(df, join: pd.DataFrame, prefix: str = '', prefix_left='', force_multi_index=False, ffill=False):
    if df is None:
        if force_multi_index:
            if isinstance(join.columns, pd.MultiIndex):
                return join
            else:
                return add_multi_index(join, prefix)
        else:
            return join

    if force_multi_index:
        if not isinstance(df.columns, pd.MultiIndex) and len(df.columns) > 0:
            if len(prefix_left) <= 0:
                raise ValueError("You need to provide a prefix_left")
            else:
                df = add_multi_index(df, prefix_left)

    if isinstance(df.columns, pd.MultiIndex) and not isinstance(join.columns, pd.MultiIndex):
        b = join.copy()
        b.columns = pd.MultiIndex.from_product([[prefix], b.columns])
        if ffill:
            return pd\
                .merge(df, b, left_index=True, right_index=True, how='outer', sort=True)\
                .fillna(method='ffill')\
                .dropna()
        else:
            return pd.merge(df, b, left_index=True, right_index=True, how='inner', sort=True)
    else:
        if ffill:
            return pd\
                .merge(df.add_prefix(prefix_left), join.add_prefix(prefix), left_index=True, right_index=True, how='outer', sort=True)\
                .fillna(method='ffill')\
                .dropna()
        else:
            return pd.merge(df.add_prefix(prefix_left), join.add_prefix(prefix), left_index=True, right_index=True, how='inner', sort=True)


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


def intersection_of_index(*dfs: pd.DataFrame):
    intersect_index = dfs[0].index

    for i in range(1, len(dfs)):
        if dfs[i] is not None:
            intersect_index = intersect_index.intersection(dfs[i].index)

    return intersect_index


def loc_if_not_none(df, value):
    if df is None:
        return None
    else:
        return df.loc[value]


def get_pandas_object(po: PandasObject, item, **kwargs):
    if item is None:
        _log.info("passed item was None")
        return None
    elif isinstance(item, Constant):
        return pd.Series(np.full(len(po), item.value), name=f"{item.value}", index=po.index)
    elif isinstance(item, PandasObject):
        return item
    else:
        if isinstance(item, Callable):
            # also allow callables where we pass kwargs and such ...
            res = call_callable_dynamic_args(item, po, **kwargs)
            if isinstance(res, PandasObject):
                return res
            else:
                return pd.Series(res, index=po.index)
        if isinstance(item, List):
            res = None
            for sub_item in item:
                sub_po = get_pandas_object(po, sub_item, **kwargs)
                if sub_po is None:
                    pass  # do nothing
                if isinstance(sub_po, pd.Series):
                    if res is None:
                        res = sub_po.to_frame()
                    else:
                        if sub_po.name in res.columns:
                            raise ValueError(f"{sub_po.name} already in {res.columns}")
                        else:
                            res = res.join(sub_po)
                elif isinstance(sub_po, pd.DataFrame):
                    if res is None:
                        res = sub_po
                    else:
                        duplicates = [col for col in sub_po.columns if col in res.columns]
                        if len(duplicates) > 0:
                            raise ValueError(f"{duplicates} already in {res.columns}")
                        else:
                            res = res.join(sub_po)
                else:
                    raise ValueError(f"Unknown pandas object {type(sub_po)}")

            return res
        else:
            try:
                if has_indexed_columns(po):
                    if item in po.columns:
                        return po[item]
                    else:
                        if isinstance(po.columns, pd.MultiIndex):
                            # try partial match
                            cols = {col: col.index(item) for col in po.columns.tolist() if item in col}

                            if len(cols) <= 0:
                                # try regex
                                cols = {col: i for col in po.columns.tolist() for i, part in enumerate(col) if re.compile(item).match(part)}

                            levels = set(cols.values())
                            if len(levels) == 1:
                                return po[cols.keys()].swaplevel(0, levels.pop(), axis=1)
                            else:
                                return po[cols.keys()]
                        else:
                            # try regex
                            return po[list(filter(re.compile(item).match, po.columns))]
                else:
                    # return po[item]
                    raise KeyError("should never get here ?")
            except KeyError:
                raise KeyError(f"{item} not found in {po.columns if hasattr(po, 'columns') else po}")

