import numpy as _np
import pandas as _pd

from pandas_ml_common import get_pandas_object as _get_pandas_object


def ta_realative_candles(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume"):
    relative = _pd.DataFrame(index=df.index)
    relative[open] = (_np.log(df[open]) - _np.log(df[close].shift(1)))
    relative[close] = (_np.log(df[close]) - _np.log(df[close].shift(1)))
    relative[high] = (_np.log(df[high]) - _np.log(df[close].shift(1)))
    relative[low] = (_np.log(df[low]) - _np.log(df[close].shift(1)))

    if volume is not None:
        relative[volume] = relative[volume].pct_change()

    return relative


def ta_candles_as_culb(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume", relative_close=False):
    o = _get_pandas_object(df, open)
    c = _get_pandas_object(df, close)
    h = _get_pandas_object(df, high)
    l = _get_pandas_object(df, low)
    oc = _pd.concat([o, c], axis=1)
    c_1 = c.shift(1)

    # calculate close, upper_shadow, lower_shadow, body
    res = _pd.DataFrame({
        "close": (c / c_1 - 1) if relative_close else c,
        "upper": (h / oc.max(axis=1) - 1),
        "lower": (oc.min(axis=1) / l - 1),
        "body": (c / o - 1)
    }, index=df.index)

    if volume is not None:
        res[volume] = df[volume].pct_change()

    return res


def ta_candle_category(df: _pd.DataFrame, open="Open", high="High", low="Low", close="Close", body_threshold=0.975, gap_thresold=0.002):
    """
    We try to classify single candle sticks based on simple rules:
        t = treshold i.e. 975%

        red, green candles:
          g: positive candle
          r: negative candle

        opening gaps
          positive opening gap (x3, x1)
          negative opening gap (x1, x3)
          no gap (x2 / x2)

        shapes:
            https://blog.quantinsti.com/candlestick-patterns-meaning/
            1. body 99 % of hl
            2. lower shadow > upper shadow && shadows > body
            3. body > upper + lower shadow
            4. shadows > body
            5. upper shadow > lower shadow  && shadows > body

    :param df: data frame containing the open, high, low, close data
    :return: integer number of category: 2 * 3 * 8 = 48
    """

    df = df.copy()
    df["close-1"] = df[close].shift(1)

    def map(df):
        o, h, l, c, c1 = df[open], df[high], df[low], df[close], df["close-1"]
        hl = (h / l - 1)
        gap = o / c1 - 1
        sign, body_upper, body_lower, rank = (1, c, o, [5, 4, 1, 2, 3]) if c > o else (-1, o, c, [-5, -1, -4, -2, -3])
        body = body_upper / body_lower - 1
        lower_shadow = body_lower / l - 1
        upper_shadow = h / body_upper - 1
        shadow = lower_shadow + upper_shadow

        if abs(gap) > gap_thresold:
            if sign > 0:
                gap = 3 if gap > 0 else 1
            else:
                gap = 3 if gap < 0 else 1
        else:
            gap = 2

        if body >= (hl * body_threshold):
            return rank[0] * gap
        else:
            if shadow > body:
                if lower_shadow > (upper_shadow + body):
                    return rank[1] * gap
                elif upper_shadow > (lower_shadow + body):
                    return rank[2] * gap
                else:
                    return rank[3] * gap
            else:
                return rank[4] * gap

    ranks = df.apply(map, raw=False, axis=1)
    return ranks.rename("candle_type")
