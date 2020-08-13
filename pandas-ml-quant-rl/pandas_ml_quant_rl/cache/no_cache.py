from typing import Callable

from pandas_ml_common import Typing
from .abstract_cache import Cache


class NoCache(Cache):

    def __init__(self, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        super().__init__(data_provider)

    def get_data_or_fetch(self, symbol):
        return self.data_provider(symbol)

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels):
        (features, _), labels, targets, weights, loss = df._.extract(features_and_labels)
        return features, labels, targets, weights, loss
