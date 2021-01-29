from typing import Union as _Union

import numpy as _np
import pandas as _pd

from pandas_ta_quant._decorators import *

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
@for_each_column
def ta_future_pct_to_current_mean(df: _pd.Series, forecast_period=1, period=14, log=False):
    future = df.shift(-forecast_period)
    mean = df.rolling(period).mean()

    return _np.log((future / mean)) if log else (future / mean - 1)

