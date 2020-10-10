"""Augment pandas DataFrame with methods to fetch time series data for quant finance"""
__version__ = '0.2.0'

import pandas as pd  # TODO obsolete
import logging
import importlib

import pandas_ml_quant_data_provider.datafetching as data_fetchers  # TODO obsolete
from .fetch import fetch_timeseries  # TODO obsolete
from .provider import *  # TODO obsolete
from .quant_data import QuantData as _QuantData


_log = logging.getLogger(__name__)
quant_data = _QuantData()


try:
    pandas_ml_common = importlib.import_module("pandas_ml_common")
    _log.warning(f"automatically imported pandas_ml_utils {pandas_ml_common.__version__}")
except:
    pass

# FIXME obsolete monkey patching ...
setattr(pd, "fetch_timeseries", fetch_timeseries)
setattr(pd, "read_ts_csv", data_fetchers.read_ts_csv)

# add data fetcher functions
for fetcher_functions in [data_fetchers]:
    for fetcher_function in dir(fetcher_functions):
        if fetcher_function.startswith("fetch_"):
            setattr(pd, fetcher_function, getattr(fetcher_functions, fetcher_function))
