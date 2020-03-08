import re
from typing import List, Callable

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject


def unpack_nested_arrays(df: pd.DataFrame):
    # get raw values
    values = df.values

    # un-nest eventually nested numpy arrays
    if values.dtype == 'object':
        if len(values.shape) > 1:
            # stack all rows per column then stack all columns
            return np.array([np.array([np.array(v) for v in values[:, col]]) for col in range(values.shape[1])]) \
                     .swapaxes(0, 1)
        else:
            # stack all rows
            return np.array([np.array(v) for v in values])
    else:
        return values


def get_pandas_object(po: PandasObject, item, **kwargs):
    if isinstance(item, PandasObject):
        return item
    else:
        if isinstance(item, Callable):
            pass
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
                            cols = [col for col in po.columns.tolist() if item in col]

                            if len(cols) <= 0:
                                # try regex
                                cols = [col for col in po.columns.tolist() for part in col if re.compile(item).match(part)]

                            return po[cols]
                        else:
                            # try regex
                            return po[list(filter(re.compile(item).match, po.columns))]
                else:
                    return po[item]
            except KeyError:
                raise KeyError(f"{item} not found in {po.columns if hasattr(po, 'columns') else po}")

