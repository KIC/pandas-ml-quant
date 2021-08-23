import logging
import traceback
from datetime import datetime, timedelta

import cachetools
import requests
import yfinance as yf
from cachier import cachier
from yfinance import utils as yf_utils, Ticker, utils

from pandas_ml_common import pd
from pandas_quant_data_provider.data_provider.time_utils import time_until_end_of_day
from pandas_quant_data_provider.symbol import Symbol

print("importing yfinance", yf.__version__)


class YahooSymbol(Symbol):

    def __init__(self, symbol: str):
        self.symbol = symbol

    def fetch_price_history(self, period: str = 'max', **kwargs):
        return _download_yahoo_data(self.symbol, period, **kwargs)

    def fetch_option_chain(self, max_maturities=None):
        return _fetch_option_chain(self.symbol, max_maturities)


def _download_yahoo_data(symbol, period, **kwargs):
    df = None

    # bloody skew index does not have any data on yahoo
    if symbol == '^SKEW':
        df = pd.read_csv('http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/skewdailyprices.csv',
                          skiprows=1,
                          parse_dates=True,
                          index_col='Date') \
            .drop(['Unnamed: 2', 'Unnamed: 3'], axis=1)
        return df
    else:
        df = _fetch_hist(symbol, period, **kwargs)
        logging.info(f'number of rows for {symbol} = {len(df)}, from {df.index[0]} to {df.index[-1]} period={period}')

        try:
            # first try to append the most recent data
            df = yf.Ticker(symbol).history(period="1d", interval='1d')[-1:].combine_first(df)
        except IOError:
            traceback.print_exc()
            logging.warning(
                'failed to add yf.Ticker({v}).history(period="1d", interval="1d")[-1:] fallback to hist only!')

        # if 'Adj Close' in df then utils.auto_adjust(df)
        return yf_utils.auto_adjust(df) if 'Adj Close' in df else df


@cachier(stale_after=time_until_end_of_day())
def _fetch_hist(symbol, period, **kwargs):
    return yf.Ticker(symbol).history(period=period, **kwargs)


#@cachetools.cached(cache=cachetools.TTLCache(maxsize=10, ttl=60))
@cachier(stale_after=timedelta(minutes=1))
def _fetch_option_chain(ticker: str, max_maturities=None):
    t = Ticker(ticker)
    expirations = t.options[:max_maturities] if max_maturities is not None else t.options

    chain_dict = {exp: t.option_chain(exp) for exp in expirations}
    frames = {}

    for exp, options in chain_dict.items():
        # ['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'contractSize', 'currency']
        calls = options.calls[['contractSymbol', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'strike']]
        puts = options.puts[['contractSymbol', 'bid', 'ask', 'lastPrice', 'impliedVolatility', 'strike']]
        df = pd.merge(calls, puts, how='outer', left_on='strike', right_on='strike', suffixes=["_call", "_put"])
        df.index = pd.MultiIndex.from_product([[exp], df["strike"]])
        frames[exp] = df

    option_chain = pd.concat(frames.values(), axis=0)
    option_chain.columns = ['call_contract', 'call_bid', 'call_ask', 'call_last', 'call_IV', 'strike',
                            'put_contract', 'put_bid', 'put_ask', 'put_last', 'put_IV']

    return option_chain.sort_index()

