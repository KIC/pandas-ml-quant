"""Augment pandas DataFrame with methods for machine learning"""
__version__ = '0.2.0'

import logging
from typing import Union, List, Callable, Any

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from pandas_ml_common.df.ml import ML
from pandas_ml_common.lazy import LazyInit
from pandas_ml_common.utils import (
    Constant,
    ReScaler,
    get_pandas_object,
    inner_join,
    has_indexed_columns,
    np_nans,
    temp_seed,
    flatten_multi_column_index,
    call_callable_dynamic_args,
    serialize,
    deserialize,
    serializeb,
    deserializeb,
    add_multi_index,
    unique_level_columns,
    unique_level_rows,
    unique_level
)

from pandas_ml_common.sampling import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")

np.nans = np_nans
setattr(PandasObject, "_", property(lambda self: ML(self)))
setattr(PandasObject, "inner_join", inner_join)
setattr(pd.DataFrame, "to_frame", lambda self: self)
setattr(pd.DataFrame, "flatten_columns", flatten_multi_column_index)
setattr(pd.DataFrame, "unique_level_columns", unique_level_columns)
setattr(pd.DataFrame, "has_indexed_columns", lambda self: has_indexed_columns(self))
setattr(pd.DataFrame, "add_multi_index", lambda self, *args, **kwargs: add_multi_index(self, *args, **kwargs))
setattr(pd.MultiIndex, "unique_level", lambda self, *args: unique_level(self, *args))
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