from typing import Union as _Union

import pandas as _pd

import pandas_ml_quant.indicators as _i

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_future_crossings(df: _PANDAS, a=None, b=None, period=1, forecast=1):
    crossings = _i.ta_cross(df, a, b, period=period)

    if forecast > 1:
        crossings = _i.ta_rnn(crossings, range(1, forecast))

    return crossings.shift(-forecast)


def ta_future_bband_quantile(df: _pd.Series, forecast_period=5, period=14, stddev=2.0, ddof=1):
    # we want to know if a future price is violating the current upper/lower band
    bands = _i.ta_bbands(df, period, stddev, ddof)
    upper = bands["upper"]
    lower = bands["lower"]
    future = df.shift(-forecast_period)
    quantile = (future > upper).astype(int) - (future < lower).astype(int)

    return quantile + 1