"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.1.1'

import logging

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from pandas_ml_common.df.ml import ML
from pandas_ml_common.lazy import LazyInit
from pandas_ml_common.utils import get_pandas_object, Constant

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "ml", property(lambda self: ML(self)))
setattr(pd.DataFrame, "to_frame", lambda self: self)
