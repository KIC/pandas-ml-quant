import logging
import traceback

import yfinance as yf
from cachier import cachier
from yfinance import utils as yf_utils

from pandas_ml_common import pd
from pandas_quant_data_provider.data_provider.time_utils import time_until_end_of_day
from pandas_quant_data_provider.symbol import Symbol


class YahooSymbol(Symbol):

    def __init__(self, symbol: str):
        self.symbol = symbol

    def get_provider_args(self):
        return [self.symbol]


def fetch_yahoo(symbol: str, period: str = 'max', **kwargs):
    return _download_yahoo_data(symbol, period, **kwargs)


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
