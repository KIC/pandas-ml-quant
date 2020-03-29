import pandas as _pd
from typing import Union as _Union

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_future_pct_to_current_mean(df: _pd.Series, forecast_period=1, period=14):
    future = df.shift(-forecast_period)
    mean = df.rolling(period).mean()

    return (future / mean) - 1

