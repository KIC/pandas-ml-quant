"""Augment pandas DataFrame with methods to fetch time series data for quant finance"""
__version__ = '0.1.7'

import pandas_ml_quant_data_provider.datafetching as data_fetchers
import pandas


# add data fetcher functions
for fetcher_functions in [data_fetchers]:
    for fetcher_function in dir(fetcher_functions):
        if fetcher_function.startswith("fetch_"):
            setattr(pandas, fetcher_function, getattr(fetcher_functions, fetcher_function))
