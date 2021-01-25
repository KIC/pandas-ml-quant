import logging
import os
from typing import List, Tuple, Iterator

import cachetools.func
import cachetools.func
from fredapi import Fred

from pandas_ml_common.utils import add_multi_index

_log = logging.getLogger(__name__)


@cachetools.func.ttl_cache(maxsize=1, ttl=10 * 60)
def fetch_fred(symbols, multi_index=False):
    if "FRED_API_KEY" not in os.environ:
        raise ValueError("you need to set the environment variable `FRED_API_KEY`. You can get a key as described here:"
                         "https://github.com/mortada/fredapi")

    fred = Fred()
    df = None

    if not isinstance(symbols, (List, Tuple, Iterator)):
        symbols = [symbols]

    for symbol in symbols:
        series = fred.get_series(symbol)

        if not series.name:
            series.name = symbol

        if df is None:
            df = series.to_frame()
        else:
            df = df.join(series)

    return add_multi_index(df, "FRED", True) if multi_index else df