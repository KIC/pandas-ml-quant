from typing import Union as _Union

import pandas as _pd

import pandas_ml_quant.indicators as _i

_PANDAS = _Union[_pd.DataFrame, _pd.Series]

def ta_future_crossings(df: _PANDAS, a=None, b=None, period=1, forecast=1):
    crossings = _i.ta_cross(df, a, b, period=period)
    _i.ta_rnn(crossings, range(1, forecast + 1))