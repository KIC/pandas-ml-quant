from .auto_regression import *
from .single_object import *
from .multi_object import *

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