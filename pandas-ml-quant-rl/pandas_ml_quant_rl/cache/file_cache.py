import os
from typing import Callable

from pandas_ml_common import pd, Typing
from pandas_ml_common.decorator import MultiFrameDecorator
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

        # note! features can be a MultiFrameDecorator !!
        if isinstance(features_and_labels.features, tuple):
            fkeys = [f'{symbol}__features__{i}' for i in range(len(features_and_labels.features))]
            features = [self.file_cache[fkey] for fk in self.file_cache if fk in fkeys]

            if len(features) == len(fkeys):
                features = MultiFrameDecorator(features)
            else:
                _, features, _ = extract_features(df, features_and_labels)
                for fk, f in zip(fkeys, features.frames()):
                    self.file_cache[fk] = f
        else:
            if fkey in self.file_cache:
                features = self.file_cache[fkey]
            else:
                print(f"fetch data for: {symbol}")
                _, features, _ = extract_features(df, features_and_labels)
                self.file_cache[fkey] = features

        return features._.values, features.index

