"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.1.0'

import sys as _sys

from pandas_ml_common import *
from pandas_ml_utils.data.extraction import *
from pandas_ml_utils.df.model import Model as DfModelExtension

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "model", property(lambda self: DfModelExtension(self)))


if 'pandas_ml_quant' in _sys.modules:
    import pandas_ml_quant
    _log.warning(f"automatically imported pandas_ml_quant {pandas_ml_quant.__version__}")
