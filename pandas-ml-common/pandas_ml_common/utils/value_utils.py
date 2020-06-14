import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def unpack_nested_arrays(df: pd.DataFrame) -> np.ndarray:
    # get raw values
    values = df.values
    res = None

    # un-nest eventually nested numpy arrays
    if values.dtype == 'object':
        _log.debug("unnesting objects, expecting numpy arrays")

        if values.ndim > 1:
            # stack all rows per column then stack all columns
            res = np.array([np.array([np.array(v) for v in values[:, col]]) for col in range(values.shape[1])]) \
                    .swapaxes(0, 1)
        else:
            # stack all rows
            res = np.array([np.array(v) for v in values])
    else:
        res = values

    if res.ndim == 3 and res.shape[1] == 1:
        return res[:, 0, :]
    else:
        return res


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
