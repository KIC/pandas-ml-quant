"""Augment pandas DataFrame with methods to fetch time series data for quant finance"""
__version__ = '0.2.0'

from .fetch import QuantDataFetcher

fetch = QuantDataFetcher().fetch