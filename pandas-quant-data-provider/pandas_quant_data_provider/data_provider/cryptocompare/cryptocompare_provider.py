import logging
import os
from datetime import datetime, timezone, timedelta
from time import sleep
from typing import Tuple, Union

import requests

from pandas_quant_data_provider.symbol import Symbol
from pandas_quant_data_provider.utils.cache import requests_cache
import pandas as pd

TSCENTER = datetime.utcnow().replace(2000, 1, 1, 0, 0, 0, 0)
MAX_TIMESTEPS = 2000


class CryptoCompareSymbol(Symbol):

    def __init__(self, frequency: Union[str, Tuple[int, str]], coin: str = None, quote: str = None):
        # separate frequency and aggregate, like 15 minute
        self.frequency = (frequency[1] if isinstance(frequency, tuple) else frequency)
        self.aggregate = (frequency[0] if isinstance(frequency, tuple) else 1)
        self.quote = quote

        # defne a generic symbol
        self.symbol = f"{coin.upper()}{quote.upper()}"

        # define the url, can be one of the following base urls
        #   https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=10&api_key=
        #   https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=10&api_key=
        #   https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=GBP&limit=10&api_key=
        self.url = f"https://min-api.cryptocompare.com/data/v2/histo{self.frequency.lower()}" \
                   f"?fsym={coin.upper()}" \
                   f"&tsym={quote.upper()}" \
                   f"&aggregate={self.aggregate}" \
                   f"&limit=2000" \
                   f"&api_key={os.environ.get('CC_API_KEY', '')}"

    def for_coins(self, *args):
        # just a helper function for multiple coins wih same frequency and quote
        return [[CryptoCompareSymbol((self.aggregate, self.frequency), a, self.quote) for a in arg]
                if isinstance(arg, (list, set, tuple)) else CryptoCompareSymbol((self.aggregate, self.frequency), arg, self.quote) for arg in args]

    def fetch_price_history(self, **kwargs) -> pd.DataFrame:
        interval_from, interval_to, timedelta_inc = self._get_current_interval()
        ts_from = int(interval_from.timestamp())
        ts_to = int(interval_to.timestamp())
        #print(ts_to, ts_from)

        # always skip caching for the first two windows in order to maintain the fixed size windows
        most_recent = CryptoCompareSymbol._fetch_raw_data(self.url + f"&toTs={ts_to}", no_cache=True)
        next_most_recent = CryptoCompareSymbol._fetch_raw_data(self.url + f"&toTs={ts_from}", no_cache=True)
        data = most_recent["Data"]["Data"]

        if sum([b["close"] for b in next_most_recent["Data"]["Data"]]) >= 1e-6:
            # note that the aggregation starts from the beginning and leaves the last aggregation as is
            # (potentially only a partial aggregation)
            data += next_most_recent["Data"]["Data"] if self.aggregate <= 1 else next_most_recent["Data"]["Data"][:-1]
            interval_to = interval_from

            # now loop all cacheable calls
            while True:
                interval_to -= timedelta_inc
                ts = int(interval_to.timestamp())
                url = self.url + f"&toTs={ts}"
                logging.debug(f"fetch url {url}")
                # print(ts, f"fetch url {url}")

                hist = CryptoCompareSymbol._fetch_raw_data(url, no_cache=False)
                if len(hist["Data"]["Data"]) <= 0 or sum([b["close"] for b in hist["Data"]["Data"]]) < 1e-6:
                    break

                # note that the aggregation starts from the beginning and leaves the last aggregation as is
                # (potentially only a partial aggregation)
                data += next_most_recent["Data"]["Data"] if self.aggregate <= 1 else next_most_recent["Data"]["Data"][:-1]

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df["time"], unit='s', origin='unix', utc=True)
        df = df.drop_duplicates().sort_index()

        return df

    @staticmethod
    @requests_cache
    def _fetch_raw_data(url: str, no_cache: bool):
        return requests.get(url).json()

    def _get_current_interval(self):
        inc = None
        if self.frequency == "minute":
            inc = timedelta(minutes=MAX_TIMESTEPS)
        if self.frequency == "hour":
            inc = timedelta(hours=MAX_TIMESTEPS)
        else:
            inc = timedelta(days=MAX_TIMESTEPS)

        t = TSCENTER
        now = datetime.utcnow()
        interval_from, interval_to = None, None

        while t < now:
            interval_from = t; t += inc; interval_to = t

        return interval_from, interval_to, inc

