import logging
import os
from datetime import datetime, timedelta
from typing import Tuple, Union

import pandas as pd
import requests

from pandas_quant_data_provider.symbol import Symbol
from pandas_quant_data_provider.utils.cache import requests_cache, Duration

TSCENTER = datetime.utcnow().replace(2000, 1, 1, 0, 0, 0, 0)
MAX_TIMESTEPS = 2000


class CryptoCompareSymbol(Symbol):

    def __init__(self, frequency: Union[str, Tuple[int, str]], coin: str = None, quote: str = None):
        # separate frequency and aggregate, like 15 minute
        self.frequency = (frequency[1] if isinstance(frequency, tuple) else frequency)
        self.aggregate = (frequency[0] if isinstance(frequency, tuple) else 1)
        self.quote = quote

        # defne a generic symbol
        self.symbol = f"{str(coin).upper()}{quote.upper()}"

        # define the url, can be one of the following base urls
        #   https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=10&api_key=
        #   https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=10&api_key=
        #   https://min-api.cryptocompare.com/data/v2/histominute?fsym=BTC&tsym=GBP&limit=10&api_key=
        self.url = f"https://min-api.cryptocompare.com/data/v2/histo{self.frequency.lower()}" \
                   f"?fsym={str(coin).upper()}" \
                   f"&tsym={quote.upper()}" \
                   f"&aggregate={self.aggregate}" \
                   f"&limit=2000"

    def for_coins(self, *args):
        # just a helper function for multiple coins wih same frequency and quote
        return [[CryptoCompareSymbol((self.aggregate, self.frequency), a, self.quote) for a in arg]
                if isinstance(arg, (list, set, tuple)) else CryptoCompareSymbol((self.aggregate, self.frequency), arg, self.quote) for arg in args]

    def fetch_price_history(self, **kwargs) -> pd.DataFrame:
        interval_from, interval_to, timedelta_inc = self._get_current_interval()
        ts_from = int(interval_from.timestamp())
        ts_to = int(interval_to.timestamp())

        # use ttl caching for the first two windows in order to maintain the fixed size windows
        most_recent = CryptoCompareSymbol._fetch_raw_data(self.url + f"&toTs={ts_to}", caching_duration=Duration.minute)
        hist = CryptoCompareSymbol._fetch_raw_data(self.url + f"&toTs={ts_from}", caching_duration=self.frequency)
        data = most_recent["Data"]["Data"]

        def has_more_data(json_data):
            return len(json_data["Data"]["Data"]) <= 0 or sum([b["close"] for b in json_data["Data"]["Data"]]) >= 1e-6

        def extract_bars(json_data):
            #return json_data["Data"]["Data"] if self.aggregate <= 1 else json_data["Data"]["Data"][:-1]
            return json_data["Data"]["Data"][:-1]

        if has_more_data(hist):
            # note that the aggregation starts from the beginning and leaves the last aggregation as is
            # (potentially only a partial aggregation)
            data += extract_bars(hist)
            interval_to = interval_from

            # now loop all cacheable calls
            while True:
                interval_to -= timedelta_inc
                ts = int(interval_to.timestamp())
                url = self.url + f"&toTs={ts}"
                logging.debug(f"fetch url {url}")

                hist = CryptoCompareSymbol._fetch_raw_data(url, caching_duration=Duration.forever)
                if not has_more_data(hist):
                    logging.info(f'last rate found for {self.symbol}: '
                                 f'{pd.to_datetime(hist["Data"]["Data"][-1]["time"], unit="s", origin="unix", utc=True)}:'
                                 f'{hist["Data"]["Data"][-1]}')
                    break

                # note that the aggregation starts from the beginning and leaves the last aggregation as is
                # (potentially only a partial aggregation)
                data += extract_bars(hist)

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df["time"], unit='s', origin='unix', utc=True)
        df = df.drop_duplicates().sort_index()
        assert not df.index.has_duplicates, f"{self.symbol} has duplicate {df.index[df.index.duplicated()]}"
        return df

    @staticmethod
    @requests_cache()
    def _fetch_raw_data(url: str, **kwargs):
        resp = requests.get(url + f"&api_key={os.environ.get('CC_API_KEY', '')}").json()

        if "RateLimit" in resp and len(resp["RateLimit"]) > 0:
            raise ValueError(f"rate limit {url}: {resp['RateLimit']}'")

        if resp["Response"].lower() != "success":
            raise ValueError(f"Unsuccessful request {url}: {resp}")

        return resp

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

    def __str__(self):
        # return f'{self.__class__.__name__}(({self.aggregate}, {self.frequency}), {self.symbol})'
        return self.symbol
