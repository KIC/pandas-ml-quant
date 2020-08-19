from typing import Callable, Tuple

from pandas_ml_common import Typing, pd, np
from pandas_ml_utils.ml.data.extraction import extract_features
from .abstract_cache import Cache


class MemCache(Cache):

    def __init__(self, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        super().__init__(data_provider)
        self.data = {}
        self.features = {}

    def get_data_or_fetch(self, symbol) -> Typing.PatchedDataFrame:
        if symbol not in self.data:
            self.data[symbol] = self.data_provider(symbol)

        return self.data[symbol]

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels) -> Tuple[np.ndarray, pd.Index]:
        if symbol not in self.features:
            _, features, _ = extract_features(df, features_and_labels)
            self.features[symbol] = (features._.values, features.index)

        return self.features[symbol]
