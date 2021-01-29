import pandas as _pd

# create convenient type hint
from pandas_ml_common import Typing as _t
from pandas_ta_quant._decorators import *


@for_each_top_level_row
def ta_volume_as_time(df: _t.PatchedPandas, volume="Volume"):
    if df.ndim > 1:
        res = df.copy()
        res.index = df[volume].cumsum()
        return res
    else:
        return _pd.Series(df.index, index=df.cumsum())

