"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.1.13'

import logging
from typing import Union, List, Callable, Any

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from pandas_ml_common.df.ml import ML
from pandas_ml_common.lazy import LazyInit
from pandas_ml_common.utils import get_pandas_object, Constant, inner_join, has_indexed_columns, nans

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")

np.nans = nans
setattr(PandasObject, "_", property(lambda self: ML(self)))
setattr(PandasObject, "inner_join", inner_join)
setattr(pd.DataFrame, "to_frame", lambda self: self)
setattr(pd.DataFrame, "has_indexed_columns", lambda self: has_indexed_columns(self))
# setattr(pd.Series, 'columns', lambda self: [self.name]) # FIXME leads to problems where we do hasattr(?, columns)


class Typing(object):
    PatchedDataFrame = pd.DataFrame
    PatchedSeries = pd.Series
    PatchedPandas = Union[PatchedDataFrame, PatchedSeries]

    DataFrame = pd.DataFrame
    Series = pd.Series
    Pandas = Union[DataFrame, Series]
    PdIndex = pd.Index
    _Selector = Union[str, List['MlGetItem'], Callable[[Any], Union[pd.DataFrame, pd.Series]], Constant]