import pandas as _pd
import numpy as _np
from typing import Union as _Union
import pandas_ml_quant.indicators as _i
from pandas_ml_common import get_pandas_object as _get_pandas_object

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_future_pct_to_current_mean(df: _pd.Series, forecast_period=1, period=14):
    future = df.shift(-forecast_period)
    mean = df.rolling(period).mean()

    return (future / mean) - 1

