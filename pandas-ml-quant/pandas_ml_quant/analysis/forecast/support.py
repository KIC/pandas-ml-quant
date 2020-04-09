import numpy as np
import pandas as pd
from pandas_ml_utils import Typing


def ta_fibbonaci_retracement(df: Typing.PatchedPandas, period=200, patience=3):
    current_min_max = (0, 0)
    most_recent_min_max = (0, 0)
    count = 0

    def call_after_reset(func):
        nonlocal most_recent_min_max
        nonlocal current_min_max
        nonlocal count

        current_min_max = (0, 0)
        most_recent_min_max = (0, 0)
        count = 0
        return func()

    def fibonacci(col, fact):
        nonlocal most_recent_min_max
        nonlocal current_min_max
        nonlocal count
        min_max = np.min(col), np.max(col)

        # as long min/max is changing because the window moves we use the currently valid min and max
        # but as soon as min and max stays stable we set this as the new currently valid min and max values
        if min_max == most_recent_min_max:
            if count > patience:
                current_min_max = min_max
                count = 0

            count += 1

        min_max_range = current_min_max[1] - current_min_max[0]
        most_recent_min_max = min_max

        return (min_max_range * fact + current_min_max[0]) if min_max_range > 0.001 else np.nan

    retracements = {"fourty":  0.382, "fitfy": 0.5, "sixty": 0.618}
    return pd.DataFrame(
        {k: call_after_reset(lambda: df.rolling(period).apply(lambda col: fibonacci(col, v), raw=True)) for k, v in
         retracements.items()},
        index=df.index
    )


def ta_trend_lines():
    # TODO implement this paper: http://www.meacse.org/ijcar/archives/128.pdf
    s = None # dataframe ...
    s1 = s.ta.rnn(3)[-200:].swaplevel(0, 1, axis=1)
    s2 = s1.ta.rescale((0, 1), digits=4).apply(lambda row: [row.min(), row.max()], raw=False, axis=1,
                                               result_type='expand')
    x = np.linspace(0, 1, len(s2))
    y = s2.values[:, 0]

    # ta = np.around(np.deg2rad(np.linspace(-90.0, 90.0, len(x))), 1)
    ta = np.deg2rad(np.linspace(-180.0, 180.0, 60))

    cos_ta = np.cos(ta)
    sin_ta = np.sin(ta)

    foo = np.vstack([x[i] * cos_ta + price * sin_ta for i, price in enumerate(y)]).T
    # foo = np.outer(cos_ta, x) + np.outer(sin_ta, y) + 1

    print(x.shape, cos_ta.shape, foo.shape)
    df_ta_rho = pd.DataFrame(foo, index=ta)
    df_ta_rho.plot(legend=None)

    y_price = s2[0].values

    def max_count(row):
        nr, cnt = np.unique(np.around(row, 2), return_counts=True)
        return nr[cnt.argmax()] if cnt.max() > 7 else 0

    max_count(foo[0])
    brr = df_ta_rho.apply(max_count, axis=1, raw=True)

    d_brr = brr[brr > 0].to_dict()

    s2[0].plot(figsize=(20, 9))

    for t, r in d_brr.items():
        y_trend = (r - x * np.cos(t)) / np.sin(t)
        x_near = np.argsort((y_trend - y_price) ** 2)[:8]

        # print(t, x_near)
        p1 = (x_near.min(), y_price[x_near.min()])
        p2 = (x_near.max(), y_price[x_near.max()])

        # plt.plot([s2.index[p1[0]], s2.index[p2[0]]], [p1[1], p2[1]], alpha=0.3)
