import os
from typing import Callable

from pandas_ml_common import pd, Typing
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
        lkey = f'{symbol}__labels'
        tkey = f'{symbol}__targets'
        swkey = f'{symbol}__sample_weights'
        glkey = f'{symbol}__gross_loss'

        if fkey in self.file_cache:
            features = self.file_cache[fkey]
            labels = self.file_cache[lkey]
            targets = self.file_cache[tkey] if tkey in self.file_cache else None
            weights = self.file_cache[swkey] if swkey in self.file_cache else None
            loss = self.file_cache[glkey] if glkey in self.file_cache else None
        else:
            print(f"fetch data for: {symbol}")
            (features, _), labels, targets, weights, loss = df._.extract(features_and_labels)
            self.file_cache[fkey] = features
            self.file_cache[lkey] = labels
            if targets is not None: self.file_cache[tkey] = targets
            if weights is not None: self.file_cache[swkey] = weights
            if loss is not None: self.file_cache[glkey] = loss

        return features, labels, targets, weights, loss

