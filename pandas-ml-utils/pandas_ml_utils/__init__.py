"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.1.9'

import sys as _sys
import importlib
from pandas_ml_common import *
from pandas_ml_utils.ml.data import *
from pandas_ml_utils.df.model import Model as DfModelExtension
from pandas_ml_utils.ml.model import *
from pandas_ml_utils.ml.data.extraction import FeaturesAndLabels

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "model", property(lambda self: DfModelExtension(self)))


if 'pandas_ml_quant' in _sys.modules:
    pandas_ml_quant = importlib.import_module("pandas_ml_quant")
    _log.warning(f"automatically imported pandas_ml_quant {pandas_ml_quant.__version__}")
