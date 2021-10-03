from typing import List, Callable, TypeVar, TYPE_CHECKING, Tuple, Union

import numpy as np
import pandas as pd
from pandas.core.base import PandasObject

from ..utils import Constant

if TYPE_CHECKING:
    from ..df_monkey_patch.df_values import MLCompatibleValues


# IMPORTANT keep this in sync with the monkey patching in the global __init__.py
class Patch(object):

    @property
    def ML(self) -> 'MLCompatibleValues':
        pass

    def inner_join(self, PandasObject) -> 'PatchedPandas':
        pass

    def has_multi_index_columns(self) -> bool:
        pass

    def has_multi_index_rows(self) -> bool:
        pass

    def to_frame(self) -> 'PatchedPandas':
        pass

    def flatten_columns(self) -> 'PatchedPandas':
        pass

    def unique_level_columns(self) -> List:
        pass

    def has_indexed_columns(self) -> bool:
        pass

    def add_multi_index(self, *args, **kwargs) -> 'PatchedPandas':
        pass


class PatchedFrame(pd.DataFrame, Patch):
    pass


class PatchedSeries(pd.Series, Patch):
    pass


class MlTypes(object):
    PatchedDataFrame = PatchedFrame
    PatchedSeries = PatchedSeries
    PatchedPandas = Union[PatchedDataFrame, PatchedSeries]
    AnyPandasObject = PandasObject

    DataFrame = pd.DataFrame
    Series = pd.Series
    PdIndex = pd.Index
    Array = np.ndarray
    DataSelector = Union[str, List['MlGetItem'], PandasObject, Constant, Callable[..., PandasObject]]

    Loss = Tuple[float, List[float]]
