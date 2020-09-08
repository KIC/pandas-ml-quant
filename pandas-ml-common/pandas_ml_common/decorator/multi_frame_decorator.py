from typing import Tuple

from pandas_ml_common.utils import intersection_of_index, call_callable_dynamic_args
from pandas_ml_common import Typing


class MultiFrameLocDecorator(object):

    def __init__(self, frames: Tuple[Typing.PatchedDataFrame]):
        self.frames = frames

    def __getitem__(self, item):
        return MultiFrameDecorator([f.loc[item] for f in self.frames])


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
        # TODO later we want the MultiFrameDecorator to also be a possible Use Case to fit a model on multiple frames of
        #  the same kind. i.e. MultiFrameDecorator([spy, gld]).model.fit(...)
        #  in this case we actually do not want the intersection of the index
        #  therefore we would also need to extend: pandas-ml-utils/pandas_ml_utils/__init__.py:16 and add some logic for
        #  fit, predict and backtest
        #  we might also want to extend the data fetchers to not only return MultiIndex frames but also MultiDecorated
        #  frames

    def frames(self, copy=True):
        if copy:
            return tuple([f.loc[self._index].copy() for f in self._frames])
        else:
            return tuple([f.loc[self._index] for f in self._frames])

    @property
    def index(self):
        return self._index

    @property
    def loc(self):
        return MultiFrameLocDecorator(self._frames)

    @property
    def _(self):
        return MultiFrameExtDecorator(self._frames)

    def __len__(self):
        return len(self.index)


