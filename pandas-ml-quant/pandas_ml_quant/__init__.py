"""Augment pandas DataFrame with methods for quant analysis"""
__version__ = '0.1.6'

import sys as _sys

import pandas

import pandas_ml_quant.data.datafetching as data_fetchers
from pandas_ml_common import *
from pandas_ml_quant.df.technical_analysis import TechnicalAnalysis as _TA
from pandas_ml_quant.model import *

_log = logging.getLogger(__name__)
_log.debug(f"numpy version {np.__version__}")
_log.debug(f"pandas version {pd.__version__}")


setattr(PandasObject, "ta", property(lambda self: _TA(self)))


if 'pandas_ml_utils' in _sys.modules:
    import pandas_ml_utils
    _log.warning(f"automatically imported pandas_ml_utils {pandas_ml_utils.__version__}")


# add read_csv short cut
setattr(pandas, "read_ts_csv", data_fetchers.read_ts_csv)

# add data fetcher functions
for fetcher_functions in [data_fetchers]:
    for fetcher_function in dir(fetcher_functions):
        if fetcher_function.startswith("fetch_"):
            setattr(pandas, fetcher_function, getattr(fetcher_functions, fetcher_function))


