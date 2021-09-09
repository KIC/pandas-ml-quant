from abc import abstractmethod
from typing import Tuple, List

import pandas as pd


class Symbol(object):

    @abstractmethod
    def fetch_price_history(self, period: str = 'max', **kwargs):
        raise NotImplemented

    def spot_price_column_name(self):
        return None

    @abstractmethod
    def fetch_option_chain(self, max_maturities=None) -> pd.DataFrame:
        """

        :param max_maturities: maximum number of expiration dates to download (None for all)
        :return: returns an option chand datafrae where the index must be a pd.MultiIndex[ExpirationDate, StrikePrice]
        """
        raise NotImplemented

    def put_columns_call_columns(self) -> Tuple[List[str], List[str]]:
        return [], []

