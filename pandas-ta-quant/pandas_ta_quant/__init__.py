"""Augment pandas DataFrame with methods for technical quant analysis"""
__version__ = '0.2.7'

import logging

from pandas.core.base import PandasObject

from pandas_ml_common import pd, np
from pandas_ta_quant.pandas_patch import TechnicalAnalysis as _TA
from pandas_ta_quant.portfolio import Portfolio

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "ta", property(lambda self: _TA(self)))
