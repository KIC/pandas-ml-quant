"""Augment pandas DataFrame with methods for quant analysis"""
__version__ = '0.2.0'

import importlib

from pandas_ml_quant.df_patching.technical_analysis import TechnicalAnalysis as _TA
from pandas_ml_quant.model import *
from pandas_ml_common import *
from pandas_ml_utils import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "ta", property(lambda self: _TA(self)))

# auto import pandas ml utils
try:
    pandas_ml_quant_data_provider = importlib.import_module("pandas_ml_quant_data_provider")
    _log.warning(f"automatically imported pandas_ml_quant_data_provider {pandas_ml_quant_data_provider.__version__}")
except:
    _log.error(f"pandas_ml_quant_data_provider module not available but needed!")

# auto import pandas quant reinforcement learning
try:
    pandas_ml_quant_rl = importlib.import_module("pandas_ml_quant_rl")
    _log.warning(f"automatically imported pandas_ml_quant_rl {pandas_ml_quant_rl.__version__}")
except:
    pass

