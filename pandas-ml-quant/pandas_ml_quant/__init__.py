"""Augment pandas DataFrame with methods for quant analysis"""
__version__ = '0.1.14.1'

import importlib
import sys as _sys

from pandas_ml_common import *
from pandas_ml_quant.df.technical_analysis import TechnicalAnalysis as _TA
from pandas_ml_quant.model import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "ta", property(lambda self: _TA(self)))

# auto import pandas ml utils
try:
    pandas_ml_utils = importlib.import_module("pandas_ml_utils")
except:
    _log.warning(f"automatically imported pandas_ml_utils {pandas_ml_utils.__version__}")

# auto import pandas ml quant data prviders
try:
    pandas_ml_quant_data_provider = importlib.import_module("pandas_ml_quant_data_provider")
    _log.warning(f"automatically imported pandas_ml_quant_data_provider {pandas_ml_quant_data_provider.__version__}")
except:
    _log.warning("pandas_ml_quant_data_provider module not avialable!")

