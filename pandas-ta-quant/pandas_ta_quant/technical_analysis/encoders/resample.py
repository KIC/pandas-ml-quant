from typing import Union as _Union

# create convenient type hint
import pandas as _pd

from pandas_ta_quant._decorators import *

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
def ta_inverse(df: _PANDAS) -> _PANDAS:
    return df.apply(lambda col: col * -1 + col.min() + col.max(), raw=True)

