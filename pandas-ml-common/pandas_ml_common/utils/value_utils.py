import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


def unpack_nested_arrays(df: pd.DataFrame):
    # get raw values
    values = df.values

    # un-nest eventually nested numpy arrays
    if values.dtype == 'object':
        _log.debug("unnesting objects, expecting numpy arrays")

        if len(values.shape) > 1:
            # stack all rows per column then stack all columns
            return np.array([np.array([np.array(v) for v in values[:, col]]) for col in range(values.shape[1])]) \
                     .swapaxes(0, 1)
        else:
            # stack all rows
            return np.array([np.array(v) for v in values])
    else:
        return values


