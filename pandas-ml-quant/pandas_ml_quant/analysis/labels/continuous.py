import pandas as _pd
import numpy as _np
from typing import Union as _Union

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_future_pct_to_current_mean(df: _pd.Series, forecast_period=1, period=14, log=False):
    future = df.shift(-forecast_period)
    mean = df.rolling(period).mean()

    return _np.log((future / mean)) if log else (future / mean - 1)

