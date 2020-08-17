import os
from typing import Callable

from pandas_ml_common import pd, Typing
from pandas_ml_utils.ml.data.extraction import extract_features
from .abstract_cache import Cache


class FileCache(Cache):

    def __init__(self, file_name: str, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        super().__init__(data_provider)
        # remove old cache file
        os.path.exists(file_name) and os.remove(file_name)
        self.file_cache = pd.HDFStore(file_name, mode='a')

    def get_data_or_fetch(self, symbol):
        if symbol in self.file_cache:
            return self.file_cache[symbol]
        else:
            df = self.data_provider(symbol)
            self.file_cache[symbol] = df
            return df

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels):
        fkey = f'{symbol}__features'

        if fkey in self.file_cache:
            features = self.file_cache[fkey]
        else:
            print(f"fetch data for: {symbol}")
            _, features, self._targets = extract_features(df, features_and_labels)
            self.file_cache[fkey] = features

        features_index = features.index
        features = features._.values
        return features, features_index

