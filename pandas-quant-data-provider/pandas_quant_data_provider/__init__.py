"""Augment pandas DataFrame with methods to fetch time series data for quant finance"""
__version__ = '0.3.0'

from .fetch import QuantDataFetcher
import pandas_quant_data_provider.utils

_qdf = QuantDataFetcher()
fetch = _qdf.fetch_price_history
fetch_price_history = fetch
fetch_option_chain = _qdf.fetch_option_chain
