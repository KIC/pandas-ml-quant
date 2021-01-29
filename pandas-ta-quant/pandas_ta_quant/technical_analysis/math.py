from typing import Callable


def ta_div(df, a: Callable, b: Callable, name=None):
    res = a(df) / b(df).values
    if name is not None:
        if res.ndims > 1:
            res.columns = name
        else:
            res.name = name

    return res