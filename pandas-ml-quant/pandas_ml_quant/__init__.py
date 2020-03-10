"""Augment pandas DataFrame with methods for quant analysis"""
__version__ = '0.1.0'

from pandas_ml_common import *
from pandas_ml_quant.df.quant import Quant
import sys as _sys

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "q", property(lambda self: Quant(self)))


if 'pandas_ml_utils' in _sys.modules:
    import pandas_ml_utils
    _log.warning(f"automatically imported pandas_ml_utils {pandas_ml_utils.__version__}")
