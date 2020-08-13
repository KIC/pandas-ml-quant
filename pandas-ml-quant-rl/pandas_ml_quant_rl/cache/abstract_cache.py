from typing import Callable, Tuple

import yfinance

from pandas_ml_common import pd, Typing


class Cache(object):

    def __init__(self, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        self.data_provider = yfinance.download if data_provider is None else data_provider

    def get_data_or_fetch(self, symbol) -> pd.DataFrame:
        pass

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels) -> Tuple[pd.DataFrame, ...]:
        pass
