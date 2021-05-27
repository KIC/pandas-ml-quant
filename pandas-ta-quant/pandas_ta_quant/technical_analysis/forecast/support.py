from typing import Tuple

import numpy as np
from sortedcontainers import SortedKeyList
from pandas_ta_quant._decorators import *
from pandas_ml_common import Typing


@for_each_top_level_row
def ta_fibbonaci_retracement(df: Typing.PatchedPandas, period=200, patience=3):
    current_min_max = [0, 0]
    most_recent_min_max = [0, 0]
    count = [0, 0]

    def call_after_reset(func):
        nonlocal most_recent_min_max
        nonlocal current_min_max
        nonlocal count

        current_min_max = [0, 0]
        most_recent_min_max = [0, 0]
        count = [0, 0]

        return func()

    def fibonacci(col, fact):
        nonlocal most_recent_min_max
        nonlocal current_min_max
        nonlocal count
        min_max = np.min(col), np.max(col)

        # as long min/max is changing because the window moves we use the currently valid min and max
        # but as soon as min and max stays stable we set this as the new currently valid min and max values
        for i in range(2):
            if min_max[i] == most_recent_min_max[i]:
                if count[i] > patience:
                    current_min_max[i] = min_max[i]
                    count[i] = 0

                count[i] += 1

        min_max_range = current_min_max[1] - current_min_max[0]
        most_recent_min_max = min_max

        valid_min_max_range = min_max_range > 0.001 and current_min_max[0] > 0.001 and current_min_max[1] > 0.001
        return (min_max_range * fact + current_min_max[0]) if valid_min_max_range else np.nan

    retracements = {"fourty":  0.382, "fitfy": 0.5, "sixty": 0.618}
    return pd.DataFrame(
        {k: call_after_reset(lambda: df.rolling(period).apply(lambda col: fibonacci(col, v), raw=True)) for k, v in
         retracements.items()},
        index=df.index
    )


@for_each_top_level_row
def ta_trend_lines(df: Typing.PatchedSeries,
                   edge_periods=3,
                   rescale_digits=4,
                   degrees=(-90, 90),
                   angles=30,
                   rho_digits=2,
                   edge_detect='mean',
                   **kwargs
                   ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from pandas_ta_quant.technical_analysis.edge_detect import EDGE_DETECTOR
    assert df.ndim == 1 or len(df.columns) == 1, "Trend lines can only be calculated on a series"

    # edge detection
    rescaled = df.ta.rescale((0, 1), digits=rescale_digits)
    edge_or_not = EDGE_DETECTOR[edge_detect](rescaled, period=edge_periods, **kwargs)

    # set up spaces
    x = np.linspace(0, 1, len(rescaled))
    y = rescaled.values.reshape(x.shape)
    edge_x_index = np.arange(0, len(rescaled))[edge_or_not != 0]
    edge_x = x[edge_or_not != 0]
    edge_y = y[edge_or_not != 0]
    thetas = np.deg2rad(np.linspace(*degrees, len(edge_x) if angles is None else angles))

    # pre compute angeles, calculate rho's
    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    if angles is None:
        # this matrix operation might be more optimized
        rhos = np.outer(cos_theta, edge_x) + np.outer(sin_theta, edge_y)
    else:
        rhos = np.vstack([edge_x[i] * cos_theta + edge_y[i] * sin_theta for i in range(len(edge_x))]).T

    # round rhos and construct a lookup table to map back from theta/rho to x/y
    rhos, time_value_lookup_table = np.around(rhos, rho_digits), {}

    for index, rho in np.ndenumerate(rhos):
        k = (thetas[index[0]], rho)
        time = df.index[edge_x_index[index[1]]]
        value = df.iloc[edge_x_index[index[1]]]

        if k in time_value_lookup_table:
            time_value_lookup_table[k].add((time, value))
        else:
            time_value_lookup_table[k] = SortedKeyList([(time, value)], key=lambda x: x[0])

    # setup the hugh space (plots nice sinusoid's
    hough_space = pd.DataFrame(rhos, index=thetas)

    # filtering
    unique_rhos = np.unique(hough_space)

    def accumulator(row):
        rhos, counts = np.unique(row, return_counts=True)
        s = pd.Series(0, index=unique_rhos)
        s[rhos] = counts
        return s

    # generate a data frame of counts with shape [angels, rhos]
    accumulated = hough_space.apply(accumulator, axis=1)

    # build lookups for filtering
    theta_indices, rho_indices = np.unravel_index(np.argsort(accumulated.values, axis=None), accumulated.shape)
    touches = []
    distances = []
    points = []

    for i in range(len(theta_indices)):
        tp = (thetas[theta_indices[i]], unique_rhos[rho_indices[i]])

        if tp in time_value_lookup_table:
            p = time_value_lookup_table[tp]
            if len(p) > 1:
                touches.append(len(p))
                distances.append(p[-1][0] - p[0][0] if len(p) > 1 else 0)
                points.append(p)

    line_lookup_table = pd.DataFrame(
        {
            "touch": touches,
            "distance": distances,
            "points": points
        },
        index=range(len(points), 0, -1)
    )

    return accumulated, line_lookup_table


@for_each_top_level_row
def ta_ohl_trend_lines(df: Typing.PatchedPandas, close="Close", high=None, low=None):
    # TODO implement this paper: http://www.meacse.org/ijcar/archives/128.pdf
    #   analog ta_trend_lines
    if df.ndim > 1:
        c = df[close]
        h = df[high] if high is not None else None
        l = df[low] if low is not None else None
    else:
        c, h, l = df, None, None

