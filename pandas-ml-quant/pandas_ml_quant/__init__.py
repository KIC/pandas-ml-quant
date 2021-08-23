"""Augment pandas DataFrame with methods for quant analysis"""
__version__ = '0.2.7'

import importlib

import pandas_ml_quant.technical_analysis_patch as qml_indicators

from pandas_ml_quant.model import *
from pandas_ml_utils import *
from pandas_ta_quant.pandas_patch import patch_indicators, TechnicalAnalysis

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")

# patch indicators
patch_indicators(TechnicalAnalysis, [qml_indicators])


# auto import pandas ml utils
try:
    pandas_ml_quant_data_provider = importlib.import_module("pandas_quant_data_provider")
    _log.warning(f"automatically imported pandas_quant_data_provider {pandas_ml_quant_data_provider.__version__}")
except:
    _log.error(f"pandas_quant_data_provider module not available but needed!")


# auto import pandas quant plotting
try:
    pandas_ta_quant_plot = importlib.import_module("pandas_ta_quant_plot")
    _log.warning(f"automatically imported pandas_ta_quant_plot {pandas_ta_quant_plot.__version__}")
except:
    pass


# auto import pandas quant reinforcement learning
try:
    pandas_ml_quant_rl = importlib.import_module("pandas_ml_quant_rl")
    _log.warning(f"automatically imported pandas_ml_quant_rl {pandas_ml_quant_rl.__version__}")
except:
    pass

