from typing import Tuple

import numpy as np

from pandas_ml_common import Typing
from pandas_ml_common.utils import intersection_of_index, call_callable_dynamic_args, add_multi_index, \
    flatten_multi_column_index
import pandas as pd


class MultiFrameLocDecorator(object):

    def __init__(self, frames: Tuple[Typing.PatchedDataFrame]):
        self.frames = frames

    def __getitem__(self, item):
        return MultiFrameDecorator([f.loc[item] for f in self.frames])


class MultiFrameILocDecorator(object):

    def __init__(self, frames: Tuple[Typing.PatchedDataFrame]):
        self.frames = frames

    def __getitem__(self, item):
        return MultiFrameDecorator([f.iloc[item] for f in self.frames])


class MultiFrameExtDecorator(object):

    def __init__(self, frames: Tuple[Typing.PatchedDataFrame]):
        self.frames = frames

    @property
    def values(self):
        return tuple([f._.values for f in self.frames])

    def extract(self, func: callable, *args, **kwargs):
        return tuple([call_callable_dynamic_args(func, f, *args, **kwargs) for f in self.frames])

    def max(self):
        return max([f._.max() for f in self.frames])


class MultiFrameDecorator(object):

    def __init__(self, frames: Tuple[Typing.PatchedDataFrame], use_index_intersection=False):
        self._frames = frames
        self._index = intersection_of_index(*frames) if use_index_intersection else frames[0].index
        
    def frames(self, copy=True):
        if copy:
            return tuple([f.loc[self._index].copy() for f in self._frames])
        else:
            return tuple([f.loc[self._index] for f in self._frames])

    def as_joined_frame(self):
        frame = pd.concat([add_multi_index(flatten_multi_column_index(f, as_string=True), i) for i, f in enumerate(self._frames)], axis=1, join='outer')
        frame.columns = frame.columns.to_list()
        return frame

    def copy(self):
        return MultiFrameDecorator([f.copy() for f in self._frames])

    @property
    def columns(self):
        tple = np.array([f.columns.to_list() for f in self._frames], dtype=object)
        return tple

    @property
    def index(self):
        return self._index

    @property
    def iloc(self):
        return MultiFrameILocDecorator(self._frames)

    @property
    def loc(self):
        return MultiFrameLocDecorator(self._frames)

    @property
    def _(self):
        return MultiFrameExtDecorator(self._frames)

    def __getitem__(self, key):
        return MultiFrameDecorator(tuple([f[key] for f in self._frames]))

    def __len__(self):
        return max([len(f) for f in self._frames])


