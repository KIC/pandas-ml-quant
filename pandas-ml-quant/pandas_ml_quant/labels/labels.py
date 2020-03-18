import pandas as _pd
import numpy as _np
from typing import Union as _Union
import pandas_ml_quant.indicators as _i
from pandas_ml_common import get_pandas_object as _get_pandas_object

_PANDAS = _Union[_pd.DataFrame, _pd.Series]



# FIXME
# FIXME                           OBSOLETE !!
# FIXME              functions either discrete or continuous !!!
# --------------------------------------------------------------


def _ta_future_multiband_bucket(df: _pd.Series, forecast_period=14, period=5, stddevs=[0.5, 1.0, 1.5, 2.0], ddof=1):
    buckets = _i.ta_multi_bbands(df, period, stddevs=stddevs, ddof=ddof)
    future = df.shift(-forecast_period)

    # return index of bucket of which the future price lies in
    def index_of_bucket(value, data):
        if _np.isnan(value) or _np.isnan(data).any():
            return _np.nan

        for i, v in enumerate(data):
            if value < v:
                return i

        return len(data)

    return buckets \
        .join(future) \
        .apply(lambda row: index_of_bucket(row[future.name], row[buckets.columns]), axis=1, raw=False) \
        .rename(df.name)

