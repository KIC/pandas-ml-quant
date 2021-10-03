from typing import Callable, Tuple

from pandas_ml_common import MlTypes
from pandas_ml_utils.ml.data.extraction import extract_features
from .abstract_cache import Cache
import numpy as np
import pandas as pd


class NoCache(Cache):

    def __init__(self, data_provider: Callable[[str], MlTypes.PatchedDataFrame] = None):
        super().__init__(data_provider)

    def get_data_or_fetch(self, symbol) -> MlTypes.PatchedDataFrame:
        return self.data_provider(symbol)

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels) -> Tuple[np.ndarray, pd.Index]:
        features, self._targets, _ = extract_features(df, features_and_labels)
        features_index = features.index
        features = features.ML.values

        return features, features_index
