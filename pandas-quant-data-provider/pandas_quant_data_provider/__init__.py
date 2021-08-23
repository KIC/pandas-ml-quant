"""Augment pandas DataFrame with methods to fetch time series data for quant finance"""
__version__ = '0.2.7'

from .fetch import QuantDataFetcher

_qdf = QuantDataFetcher()
fetch = _qdf.fetch_price_history
fetch_price_history = fetch
fetch_option_chain = _qdf.fetch_option_chain
