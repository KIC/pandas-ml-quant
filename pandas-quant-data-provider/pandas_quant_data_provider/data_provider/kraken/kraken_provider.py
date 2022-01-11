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


class KrakenSymbol(Symbol):

    def __init__(self, symbol: str):
        self.symbol = symbol

    def spot_price_column_name(self):
        return "Close"

    def fetch_price_history(self, period: str = 'max', **kwargs):
        return _download_yahoo_data(self.symbol, period, **kwargs)

    def fetch_option_chain(self, max_maturities=None):
        return _fetch_option_chain(self.symbol, max_maturities)

    def put_columns_call_columns(self):
        return ['put_bid', 'put_ask', 'put_last'], ['call_bid', 'call_ask', 'call_last']
