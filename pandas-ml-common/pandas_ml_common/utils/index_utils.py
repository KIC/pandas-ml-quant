import logging
import re
from collections import OrderedDict
from typing import List, Union, Callable

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args
from pandas_ml_common.utils.types import Constant

_log = logging.getLogger(__name__)


def concat_indices(indices: List[pd.Index]):
    if indices is None or len(indices) < 2:
        return indices

    idx = indices[0] if isinstance(indices[0], pd.Index) else pd.Index(indices[0])

    for i in range(1, len(indices)):
        idx = idx.append(indices[i] if isinstance(indices[i], pd.Index) else pd.Index(indices[i]))

    return idx


def has_indexed_columns(po: PandasObject):
    return hasattr(po, "columns") and isinstance(po.columns, pd.Index)


def flatten_multi_column_index(df: pd.DataFrame, as_string=False):
    if df.ndim > 1:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [", ".join(col) for col in df.columns.tolist()] if as_string else df.columns.tolist()

    return df


def add_multi_index(df, head, inplace=False, axis=1, level=0):
    df = df if inplace else df.copy()

    if axis == 0:
        df.index = pd.MultiIndex.from_tuples([(head, *(t if isinstance(t, tuple) else (t, ))) for t in df.index.tolist()]).swaplevel(0, level)
    elif axis == 1:
        if df.ndim > 1:
            df.columns = pd.MultiIndex.from_tuples([(head, *(t if isinstance(t, tuple) else (t, ))) for t in df.columns.tolist()]).swaplevel(0, level)
        else:
            df.name = (head, df.name)
    else:
        raise ValueError("illegal axis, expected 0|1")

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

    if len(join) <= 0:
        _log.warning(f"right dataframe is empty: {prefix}")

    if force_multi_index:
        if not isinstance(join, pd.DataFrame):
            join = join.to_frame()

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


def same_columns_after_level(df: pd.DataFrame, level=0):
    if df.ndim < 2:
        return None

    df = df[:1].copy()
    top_level_columns = unique_level_columns(df, level)
    last_columns = None
    for tlc in top_level_columns:
        xs = df.xs(tlc, axis=1, level=level)
        this_columns = xs.columns.to_list()
        if last_columns is None or last_columns == this_columns:
            last_columns = this_columns
        else:
            return False

    return True


def unique_level_columns(df: pd.DataFrame, level=0):
    idx = df if isinstance(df, pd.Index) else df.columns
    return unique_level(idx, level)


def unique_level_rows(df: Union[pd.DataFrame, pd.Index], level=0):
    idx = df if isinstance(df, pd.Index) else df.index
    return unique_level(idx, level)


def unique_level(idx: pd.Index, level=0):
    return unique(idx.get_level_values(level)) if isinstance(idx, pd.MultiIndex) else idx


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

    return intersect_index.sort_values()


def loc_if_not_none(df, value):
    if df is None:
        return None
    else:
        return df.loc[value]


def get_pandas_object(po: PandasObject, item, type_map=None, **kwargs):
    if item is None:
        _log.info("passed item was None")
        return None
    elif isinstance(item, Constant):
        return pd.Series(np.full(len(po), item.value), name=f"{item.value}", index=po.index)
    elif type_map is not None and type(item) in type_map:
        return call_callable_dynamic_args(type_map[type(item)], po, item, **kwargs)
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
                sub_po = get_pandas_object(po, sub_item, type_map, **kwargs)
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

