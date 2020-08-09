"""Augment pandas DataFrame with methods to fetch time series data for quant finance"""
__version__ = '0.1.13'

import pandas as pd

import pandas_ml_quant_data_provider.datafetching as data_fetchers
from .fetch import fetch_timeseries
from .provider import *

setattr(pd, "fetch_timeseries", fetch_timeseries)
setattr(pd, "read_ts_csv", data_fetchers.read_ts_csv)

# add data fetcher functions
for fetcher_functions in [data_fetchers]:
    for fetcher_function in dir(fetcher_functions):
        if fetcher_function.startswith("fetch_"):
            setattr(pd, fetcher_function, getattr(fetcher_functions, fetcher_function))
