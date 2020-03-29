from typing import Union as _Union

# create convenient type hint
import numpy as _np
import pandas as _pd
from scipy.stats import zscore

from pandas_ml_common.utils import has_indexed_columns
from pandas_ml_quant.analysis.filters import ta_ema, ta_wilders, ta_sma
from pandas_ml_quant.utils import wilders_smoothing as _ws, with_column_suffix as _wcs

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


def ta_inverse(df: _PANDAS) -> _PANDAS:
    return df.apply(lambda col: col * -1 + col.min() + col.max(), raw=True)

