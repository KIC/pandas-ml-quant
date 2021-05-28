# create convenient type hint

import numpy as np
import pandas as pd

from pandas_ml_common import Typing as _t
from pandas_ta_quant._decorators import *
from pandas_ta_quant._utils import rolling_apply


@for_each_top_level_row
@for_each_column
def ta_naive_edge_detect(df: _t.PatchedDataFrame, period=3):
    assert period % 2 > 0, "only odd periods are allowed"
    center = (period - 1) // 2

    return df.rolling(period, center=True)\
             .apply(lambda s: 1 if (s[0] < s[center] > s[-1]) else -1 if (s[0] > s[center] < s[-1]) else 0, raw=True)\
             .rename(f'{df.name}_edge_naive_{period}')


@for_each_top_level_row
@for_each_column
def ta_edge_detect_mean(df: _t.PatchedSeries, period=3):
    assert df.ndim == 1 or len(df.columns) == 1, "Trend lines can only be calculated on a series"
    assert period > 2, "minimum period is 3"

    def edge(col):
        mean = col.mean()
        if col[0] > mean and col[-1] > mean:
            return 1
        elif col[0] < mean and col[-1] < mean:
            return -1
        else:
            return 0

    return df.rolling(period, center=True).apply(edge, raw=True).rename(f'{df.name}_edge_mean_{period}')


@for_each_top_level_row
@for_each_column
def ta_edge_detect_poly(df: _t.PatchedSeries, period=3, a_threshold=5, vx_threshold=0.1):
    x = np.linspace(0, 1, period)
    ab = rolling_apply(df, period, lambda s: np.polyfit(x, s.values, deg=2)[:2], ["a", "b"], center=True)

    # x of vertex will be -b/2a
    ab["vx"] = ab.apply(lambda r: -r["b"] / (2 * r["a"]), axis=1)

    if a_threshold is not None:
        ab["a"] = ab["a"].apply(lambda x: x if np.abs(x) >= a_threshold else 0)

    if vx_threshold is not None:
        # a perfect turning point would be at x=1/2
        ab["a"] = ab[["a", "vx"]].apply(lambda x: x["a"] if np.abs(x["vx"] - 0.5) <= vx_threshold else 0, axis=1)

    return df.to_frame()[[]].join(ab["a"].rename(f'{df.name}_edge_poly_{period}'))


EDGE_DETECTOR = {
    "mean": ta_edge_detect_mean,
    "naive": ta_naive_edge_detect,
    "poly": ta_edge_detect_poly,
}
