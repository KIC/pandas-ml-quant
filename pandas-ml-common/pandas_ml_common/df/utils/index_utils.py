import logging
import re
from collections import OrderedDict, Callable
from typing import List

import pandas as pd
from pandas.core.base import PandasObject

from pandas_ml_common.callable_utils import call_callable_dynamic_args

_log = logging.getLogger(__name__)


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


def get_pandas_object(po: PandasObject, item, **kwargs):
    if item is None:
        _log.warning("passed item was None")
        return None
    elif isinstance(item, PandasObject):
        return item
    else:
        if isinstance(item, Callable):
            # also allow callables where we pass kwargs and such ...
            res = call_callable_dynamic_args(item, po, **kwargs)
            assert isinstance(res, PandasObject), "callable needs to return a pandas object (series or dataframe)"
            return res
        if isinstance(item, List):
            res = None
            for sub_item in item:
                sub_po = get_pandas_object(po, sub_item)
                if isinstance(sub_po, pd.Series):
                    res = sub_po.to_frame()  if res is None else res.join(sub_po)
                elif isinstance(sub_po, pd.DataFrame):
                    res = sub_po if res is None else res.join(sub_po)
                else:
                    raise ValueError(f"Unknown pandas object {type(sub_po)}")

            return res
        else:
            try:
                if hasattr(po, 'columns'):
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
                    return po[item]
            except KeyError:
                raise KeyError(f"{item} not found in {po.columns if hasattr(po, 'columns') else po}")

