import logging
from typing import List, Union, Any

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from pandas_ml_common.utils.index_utils import unique_level_rows

_log = logging.getLogger(__name__)


def unpack_nested_arrays(df: Union[pd.DataFrame, pd.Series, np.ndarray], split_multi_index_rows=True, dtype=None) -> Union[List[np.ndarray], np.ndarray]:
    if df is None:
        return None
    elif isinstance(df, PandasObject):
        # in case of multiple assets stacked on top of each other
        if split_multi_index_rows and isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 1:
            return [unpack_nested_arrays(df.loc[group], split_multi_index_rows, dtype) for group in unique_level_rows(df)]
        else:
            values = df.values
    elif not isinstance(df, np.ndarray):
        values = np.array(df)
    else:
        values = df

    # un-nest eventually nested numpy arrays
    res = None
    if values.dtype == 'object':
        _log.debug("unnesting objects, expecting numpy arrays")

        if values.ndim > 1:
            # stack all rows per column then stack all columns
            column_values = [np.array([np.array(v, dtype=dtype) for v in values[:, col]]) for col in range(values.shape[1])]
            res = np.array(column_values).swapaxes(0, 1)  # ignore warning as this throws an exception anyways
        else:
            # stack all rows
            res = np.array([np.array(v) for v in values])
    else:
        res = values

    if res.ndim == 3 and res.shape[1] == 1:
        res = res[:, 0, :]
    if res.ndim == 3 and res.shape[-1] == 1:
        res = res[:, :, 0]

    return res


def wrap_row_level_as_nested_array(df: pd.DataFrame, row_level=-1, column_name=None, dropna=True):
    if not isinstance(df.index, pd.MultiIndex):
        return df

    if dropna: df = df.dropna()

    queries = {i[:row_level] + (i[row_level + 1:] if len(i[row_level:]) > 1 else ()) for i in df.index}
    column_name = ", ".join([str(c) for c in df.columns]) if column_name is None else column_name
    column = np.empty(len(queries), dtype=object)

    for i, query in enumerate(queries):
        column[i] = df.loc[query].values.tolist()

    res = pd.DataFrame({column_name: column}, index=queries)
    if res.index.nlevels == 1:
        res.index = [i[0] for i in res.index]

    return res


def hexplode(df: pd.DataFrame, col_name: Any, new_columns) -> pd.DataFrame:
    t = pd.DataFrame(df[col_name].tolist(), columns=new_columns, index=df.index)
    return pd.concat([df.drop(col_name, axis=1), t], axis=1)


def to_pandas(arr, index, columns) -> pd.DataFrame:
    # array has an expected shape of (rows, samples, columns, value(s))
    # remove last dimension of arr if it equals 1
    if arr.shape[-1] == 1:
        arr = arr.reshape(arr.shape[:-1])

    # initial variables
    df = pd.DataFrame({}, index=index)
    nr_columns = len(columns)
    nr_rows = len(index)

    if arr.ndim > 2:
        sample_index = 0 if arr.shape[1] == 1 else list(range(arr.shape[1]))

        # array has the shape of (rows, samples, columns, value(s))
        # each column contains embedded array
        # fill all columns up to the last one normally
        nans = np.ones((*arr.shape[:3], *arr.shape[4:])) * np.nan
        for i, col in enumerate(columns[:-1]):
            # if i < nr_columns else np.nan's in the shape of the arr
            _arr = arr[:, sample_index, i] if i < arr.shape[2] else nans[:, sample_index]
            df[col] = [row.tolist() for row in _arr] if _arr.ndim > 1 else _arr

        # fill last column
        if arr.shape[2] > nr_columns:
            # compound overflow into last column
            _arr = arr[:, sample_index, nr_columns - 1:]
            df[columns[-1]] = [row.tolist() for row in _arr] if _arr.ndim > 1 else _arr
        else:
            # fill normal last column
            _arr = arr[:, sample_index, -1]
            df[columns[-1]] = [row.tolist() for row in _arr] if _arr.ndim > 1 else _arr

    elif arr.ndim > 1:
        # array has the shape of (rows, samples)
        if nr_columns == 1:
            # one sample per column
            df[columns[0]] = [row.tolist() for row in arr] if arr.shape[1] > 1 else arr
        else:
            # fill all columns up to the last one regularly
            for i, col in enumerate(columns[:-1]):
                df[col] = arr[:, i] if i < nr_columns else np.nan

            # fill last column
            if arr.shape[1] > nr_columns:
                # if we have more data then columns pack everything into last column
                df[columns[-1]] = [row.tolist() for row in arr[:, nr_columns - 1:]] if arr.shape[1] > nr_columns else arr[:, -1]
            elif arr.shape[1] == nr_columns:
                df[columns[-1]] = arr[:, -1]
            else:
                # if we have more columns then data, we set the last column to be nan
                df[columns[-1]] = np.nan

    elif arr.ndim > 0:
        # array has the shape of (rows) and we need nr_columns to be 1 and arr.shape[0] to be nr_rows
        if arr.shape[0] != nr_rows:
            raise ValueError(f"incompatible shapes: columns={columns} ({nr_columns}), rows={nr_rows}, {arr.shape}")

        for i, col in enumerate(columns):
            # if nr of columns > 1 then fill rest with nan
            df[col] = arr if i <= 0 else np.nan
    else:
        # array is a scalar
        # we need nr_columns to be 1 and nr_rows to be 1
        if nr_rows != 1:
            raise ValueError(f"incompatible shapes: columns={columns} ({nr_columns}), rows={nr_rows}, {arr.shape}")

        for i, col in enumerate(columns):
            # if nr of columns > 1 then fill rest with nan
            df[col] = arr if i <= 0 else np.nan

    return df


def as_list(value):
    return value if value is None or isinstance(value, (tuple, list)) else [value]


def get_correlation_pairs(df: pd.DataFrame):
    corr = df.corr().abs()

    # eventually flatten MultiIndex
    if isinstance(corr.columns, pd.MultiIndex):
        corr.columns = corr.columns.tolist()
        corr.index = corr.index.tolist()

    redundant = {(df.columns[i], df.columns[j]) for i in range(0, df.shape[1]) for j in range(0, i + 1)}
    sorted_corr = corr.unstack().drop(labels=redundant).sort_values()
    return corr, sorted_corr


def cumcount(s: pd.Series):
    last_value = None
    count = 1
    res = {}

    for i, v in s.iteritems():
        if last_value == v:
            count += 1
        else:
            res[i] = count
            last_value = v
            count = 1

    return pd.Series(res, name=f'cumcount({s.name})')
