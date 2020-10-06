import logging
import math
import traceback
from datetime import datetime, timedelta
from time import sleep
from typing import Set

import cachetools
import pandas as pd
import requests
from pangres import upsert
from ytd.compat import quote, text

from pandas_ml_quant_data_provider.quant_data import DataProvider

_log = logging.getLogger(__file__)


class Yahoo(DataProvider):

    def __init__(self):
        super().__init__(__file__)

    @cachetools.func.ttl_cache(maxsize=10000, ttl=10 * 60)
    def has_symbol(self, symbol: str, **kwargs):
        with self.engine.connect() as con:
            symbols = list(con.execute(f"SELECT * FROM {DataProvider.symbols_table_name} WHERE LOWER(symbol)='{symbol.lower()}'"))
            if len(symbols) == 1:
                True
            elif len(symbols) > 1:
                _log.warning(f"Ambiguous {symbol} found more then once!\n{symbols}")
                return True

        # fall back
        return requests.get(f"https://finance.yahoo.com/quote/{symbol}", allow_redirects=False).status_code == 200

    @cachetools.func.ttl_cache(maxsize=100, ttl=10 * 60)
    def load(self, symbol: str, **kwargs):
        # TODO add some SQLite magic here
        start_date = None  # select max date from .... where symbol = symbol
        return Yahoo.__download_yahoo_data(symbol, start_date)

    def update_symbols(self, **kwargs):
        with self.engine.connect() as con:
            symbols = set(con.execute(f'SELECT * FROM {DataProvider.symbols_table_name}'))

        ticker_finder = TickerFinder(symbols)
        for symbols_df in ticker_finder.fetch():
            upsert(self.engine, symbols_df, DataProvider.symbols_table_name, if_row_exists='ignore')

    def update_quotes(self, **kwargs):
        pass

    @staticmethod
    def __download_yahoo_data(symbol, period: str = 'max', start_date: datetime = None):
        import yfinance as yf

        start = (start_date - timedelta(days=1)).strftime('%Y-%m-%d') if start_date is not None else None
        ticker = yf.Ticker(symbol)

        try:
            # first try to append the most recent data
            df = ticker.history(period="1d", interval='1d', start=start)[-1:].combine_first(ticker.history(period=period))
        except IOError:
            traceback.print_exc()
            _log.warning(
                'failed to add yf.Ticker({v}).history(period="1d", interval="1d")[-1:] fallback to hist only!')
            df = ticker.history(period=period)

        # print some statistics
        _log.info(f'number of rows for {symbol} = {len(df)}, from {df.index[0]} to {df.index[-1]} period={period}')

        # fix some historic artefacts
        if "Adj Close" in df:
            ratio = (df["Adj Close"] / df["Close"]).fillna(1)
            df = df.drop("Adj Close", axis=1).apply(lambda x: x * ratio, axis=0)

        return df


class TickerFinder(object):
    # inspired by: https://github.com/Benny-/Yahoo-ticker-symbol-downloader

    user_agent = 'yahoo-ticker-symbol-downloader'
    general_search_characters = 'abcdefghijklmnopqrstuvwxyz0123456789.='
    first_search_characters = 'abcdefghijklmnopqrstuvwxyz'
    headers = {'User-agent': user_agent}
    query_string = {'device': 'console', 'returnMeta': 'true'}
    illegal_tokens = ['null']
    max_retries = 4

    def __init__(self, existing_symbols: Set[str]):
        self.rsession = requests.Session()
        self.existing_symbols = existing_symbols
        self.queries = []

    def fetch(self):
        self._add_queries()

        for query in self.queries:
            df = self._next_request(query)
            yield df

    def _add_queries(self, prefix=''):
        if len(prefix) == 0:
            search_characters = TickerFinder.first_search_characters
        else:
            search_characters = TickerFinder.general_search_characters

        for i in range(len(search_characters)):
            element = str(prefix) + str(search_characters[i])
            if element not in TickerFinder.illegal_tokens and element not in self.existing_symbols and element not in self.queries:
                self.queries.append(element)

    def _fetch(self, query_str):
        def _encodeParams(params):
            encoded = ''
            for key, value in params.items():
                encoded += ';' + quote(key) + '=' + quote(text(value))
            return encoded

        params = {
            'searchTerm': query_str,
        }

        protocol = 'https'
        req = requests.Request('GET',
                               protocol +'://finance.yahoo.com/_finance_doubledown/api/resource/searchassist' + _encodeParams(params),
                               headers=TickerFinder.headers,
                               params=TickerFinder.query_string
                               )

        req = req.prepare()
        print("req " + req.url)
        resp = self.rsession.send(req, timeout=(12, 12))
        resp.raise_for_status()

        return resp.json()

    def _next_request(self, query_str):
        def decode_symbols_container(json):
            df = pd.DataFrame(json['data']['items'])
            return df, len(df)

        success = False
        retry_count = 0
        json = None

        # Eponential back-off algorithm
        # to attempt 5 more times sleeping 5, 25, 125, 625, 3125 seconds
        # respectively.
        while(success == False):
            try:
                json = self._fetch(query_str)
                success = True
            except (requests.HTTPError,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as ex:
                if retry_count < TickerFinder.max_retries:
                    attempt = retry_count + 1
                    sleep_amt = int(math.pow(5, attempt))
                    print(f"Retry attempt: {attempt} of {TickerFinder.max_retries}. Sleep period: {sleep_amt} seconds.")
                    sleep(sleep_amt)
                    retry_count = attempt
                else:
                    raise

        (symbols, count) = decode_symbols_container(json)

        if (count >= 10):
            self._add_queries(query_str)

        return symbols


