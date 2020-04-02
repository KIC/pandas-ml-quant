import pandas as _pd

# create convenient type hint
from pandas_ml_common import Typing as _t


def ta_volume_as_time(df: _t.PatchedPandas, volume="Volume"):
    if df.has_indexed_columns():
        res = df.copy()
        res.index = df[volume].cumsum()
        return res
    else:
        return _pd.Series(df.index, index=df.cumsum())


def ta_volume_interpolated():
    # TODO ...
    # first need a line from one (volume, price) to the next (volume, price)
    # then we can do so as if we would have evenly spaced volumes and calculate the interpolated price
    # finally we can map back the interpolated price to time
    pass
