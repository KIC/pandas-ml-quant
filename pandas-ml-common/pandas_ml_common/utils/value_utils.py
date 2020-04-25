import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def unpack_nested_arrays(df: pd.DataFrame) -> np.ndarray:
    # get raw values
    values = df.values

    # un-nest eventually nested numpy arrays
    if values.dtype == 'object':
        _log.debug("unnesting objects, expecting numpy arrays")

        if values.ndim > 1:
            # stack all rows per column then stack all columns
            return np.array([np.array([np.array(v) for v in values[:, col]]) for col in range(values.shape[1])]) \
                     .swapaxes(0, 1)
        else:
            # stack all rows
            return np.array([np.array(v) for v in values])
    else:
        return values


def to_pandas(arr, index, columns) -> pd.DataFrame:
    # if the last dimension is 1 we can remove it
    if arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr.reshape(arr.shape[:-1])

    # if we have one column and a 1D vector just return a new series
    if len(columns) == 1 and arr.ndim == 1:
        return pd.DataFrame({columns[0]: arr}, index=index)

    # make sure columns is of type list
    if not isinstance(columns, list):
        columns = columns.tolist()

    # create the data frame
    df = pd.DataFrame({}, index=index)
    last_idx = len(columns) - 1

    # TODO add logic for multi index
    for i in range(len(columns)):
        if i >= last_idx and arr.shape[-1] > len(columns):
            col_vals = np.take(arr, range(i, arr.shape[-1]), -1)
        else:
            col_vals = np.take(arr, i, axis=-1)

        # might save one dimension
        if col_vals.ndim > 1 and col_vals.shape[1] == 1:
            col_vals = col_vals.reshape((arr.shape[0], *col_vals.shape[2:]))

        # eventually we need to nest back multi dimensional arrays
        df[columns[i]] = [row.tolist() for row in col_vals] if col_vals.ndim > 1 else col_vals

    return df
