import numpy as np
import pandas as pd
from numba import guvectorize, float32, int32, float64, int64

from pandas_ml_common.utils import ReScaler


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


def returns_to_log_returns(returns):
    return np.log(1 + returns)


def log_returns_to_returns(log_returns):
    return (np.e ** log_returns) - 1


@guvectorize([(float32[:], int32, float32[:]),
              (float64[:], int64, float64[:])], '(n),()->(n)')
def wilders_smoothing(arr: np.ndarray, period: int, res: np.ndarray):
    assert period > 0
    alpha = (period - 1) / period
    beta = 1 / period

    res[0:period] = np.nan
    res[period - 1] = arr[0:period].mean()
    for i in range(period, len(arr)):
        res[i] = alpha * res[i-1] + arr[i] * beta


def _rescale(df: pd.DataFrame, range=(-1, 1), digits=None, axis=None):
    if axis is not None:
        return df.apply(lambda x: _rescale(x, range), raw=False, axis=axis, result_type='broadcast')
    else:
        rescaler = ReScaler((df.values.min(), df.values.max()), range)
        rescaled = rescaler(df.values)

        if digits is not None:
            rescaled = np.around(rescaled, digits)

        if rescaled.ndim > 1:
            return pd.DataFrame(rescaled, columns=df.columns, index=df.index)
        else:
            return pd.Series(rescaled, name=df.name, index=df.index)



# TODO eventually turn this into a decorator ???
def with_column_suffix(suffix, po, ref_po=None):
    if ref_po is None:
        ref_po = po

    if po.ndim > 1:
        if isinstance(po.columns, pd.MultiIndex):
            po.columns = pd.MultiIndex.from_tuples([(f'{col[0]}_{suffix}', *col[1:]) for col in ref_po.columns.to_list()])
            return po
        else:
            po.columns = ref_po.columns
            return po.add_suffix(f'_{suffix}')
    else:
        if isinstance(po.name, tuple):
            return po.rename((suffix, *ref_po.name))
        else:
            return po.rename(f'{ref_po.name}_{suffix}')


# return index of bucket of which the future price lies in
def index_of_bucket(value, data):
    if np.isnan(value) or np.isnan(data).any() or np.isinf(value) or np.isinf(value).any():
        return np.nan

    for i, v in enumerate(data):
        if value < v:
            return i

    return len(data)


def has_decorator(function, decorator):
    # If we have no func_closure, it means we are not wrapping any other functions.
    if not function.func_closure:
        return False

    for closure in function.func_closure:
        if has_decorator(closure.cell_contents):
            return True

    return [function]


def rolling_apply(df, period, func, names, center=False, **kwargs):
    assert center is False or period % 2 > 0, "only odd periods are allowed"
    margin = (period - 1) // 2 if center else 0
    index = df.index[period - margin - 1: len(df) - margin]

    def fi(i):
        return i - period + margin

    def ti(i):
        return i + margin

    # if period is 3 dann [0:3]
    data = np.array([func(df.iloc[fi(i):ti(i)], **kwargs) for i in range(period-margin, len(df)+1-margin)])
    s = pd.Series(data, index=index, name=names) if data.ndim < 2 else pd.DataFrame(data, index=index, columns=names)

    if df.ndim < 2:
        df = df.to_frame()

    return df[[]].join(s)