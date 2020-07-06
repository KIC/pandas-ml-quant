import pandas as pd
import numpy as np


def sort_distance(s1: pd.Series, s2: pd.Series, top_percent=None) -> np.ndarray:
    s = (s1 / s2 - 1).dropna().abs()
    distances = s.sort_values().values

    if top_percent is not None:
        return distances[:int(np.ceil(len(distances) * top_percent))]
    else:
        return distances


def symmetric_quantile_factors(a, bins=11):
    return np.linspace(1-a, 1+a, bins)


def conditional_func(s: pd.Series, s_true: pd.Series, s_false: pd.Series):
    df = s.to_frame("CONDITION").join(s_true.rename("TRUTHY")).join(s_false.rename("FALSY"))
    return df.apply(lambda r: r["TRUTHY"] if r["CONDITION"] is True else r["FALSY"] , axis=1)


def difference(a: pd.Series, b: pd.Series, relative: bool, replace_inf=0):
    if relative:
        if replace_inf is not None:
            return (a / b - 1).replace([np.inf, -np.inf], replace_inf)
        else:
            return a / b - 1
    else:
        return a - b