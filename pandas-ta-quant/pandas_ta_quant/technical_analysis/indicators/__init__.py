from pandas_ml_common import pd as _pd, suitable_kwargs as _select_kwargs
from pandas_ta_quant._decorators import *
from .multi_object import *
from .single_object import *
from .time import *
from ..encoders.candles import ta_candle_category, ta_candles_as_culb


# this is executing all technical indicators into one huge data frame
@for_each_top_level_row
@for_each_top_level_column
def ta_all(df: _pd.DataFrame,
           close_only: bool = False,
           open: str = "Open",
           high: str = "High",
           low: str = "Low",
           close: str = "Close",
           volume: str = "Volume",
           open_interest: str = "Open Interest"
           ):
    assert isinstance(df, _pd.DataFrame), "Data needs to be a frame not a Series"
    df_res = None

    # loop all indicator functions
    for indicator_function in dir(single_object):
        if indicator_function.startswith("ta_"):
            func = getattr(single_object, indicator_function)

            if hasattr(func, '_is_timeconsuming'):
                continue

            if df_res is None:
                df_res = func(df[close]).to_frame().add_multi_index(indicator_function[3:])
            else:
                df_res = df_res.inner_join(func(df[close]), prefix=indicator_function[3:], force_multi_index=True)

    for indicator_function in dir(time):
        if indicator_function.startswith("ta_"):
            func = getattr(time, indicator_function)

            if hasattr(func, '_is_timeconsuming'):
                continue

            df_res = df_res.inner_join(func(df), prefix=indicator_function[3:], force_multi_index=True)

    if close_only:
        return df_res

    for indicator_function in dir(multi_object):
        if indicator_function.startswith("ta_"):
            func = getattr(multi_object, indicator_function)

            if hasattr(func, '_is_timeconsuming'):
                continue

            _df = func(df, **_select_kwargs(func, open=open, high=high, low=low, close=close, volume=volume,
                                            open_interest=open_interest))
            df_res = df_res.inner_join(_df, prefix=indicator_function[3:], force_multi_index=True)

    # add corner cases i.e. candle stock encodings as indicators
    df_res = df_res.inner_join(ta_candle_category(df, open=open, high=high, low=low, close=close), prefix='candle_category', force_multi_index=True)
    df_res = df_res.inner_join(ta_candles_as_culb(df, open=open, high=high, low=low, close=close, volume=volume), prefix='culb', force_multi_index=True)

    return df_res


"""
this module basically re-implements all oscillators from TA-Lib:
  https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
"""

"""
TODO add this missing indicators

AROONOSC - Aroon Oscillator
real = AROONOSC(high, low, timeperiod=14)
Learn more about the Aroon Oscillator at tadoc.org.

CMO - Chande Momentum Oscillator
real = CMO(close, timeperiod=14)
Learn more about the Chande Momentum Oscillator at tadoc.org.

MFI - Money Flow Index
real = MFI(high, low, close, volume, timeperiod=14)
Learn more about the Money Flow Index at tadoc.org.

STOCH - Stochastic
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
Learn more about the Stochastic at tadoc.org.

"""