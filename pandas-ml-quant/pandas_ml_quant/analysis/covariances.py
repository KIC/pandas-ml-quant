# create convenient type hint
import numpy as _np
import pandas as _pd

from pandas_ml_common import Typing


def _normalize(df, convert_to):
    data = df.copy()

    if convert_to == 'returns':
        data = df.pct_change()
    if convert_to == 'log-returns':
        data = _np.log(df) - _np.log(df.shift(1))

    return data


def ta_ewma_covariance(df: Typing.PatchedPandas, alpha=0.97, convert_to='returns', **kwargs):
    data = _normalize(df, convert_to)
    data.columns = data.columns.to_list()
    return data.ewm(com=alpha).cov()


def ta_moving_covariance(df: Typing.PatchedPandas, period=30, convert_to='returns', **kwargs):
    data = _normalize(df, convert_to)
    data.columns = data.columns.to_list()
    return data.rolling(period).cov()


def ta_sparse_covariance(df: Typing.PatchedPandas, convert_to='returns', covariance='ewma', cov_arg=0.97, rho=0.1, inverse=False, **kwargs):
    from sklearn.covariance import graphical_lasso

    if covariance in ['ewma', 'weighted']:
        cov_func = ta_ewma_covariance
    elif covariance in ['rolling', 'moving']:
        cov_func = ta_moving_covariance
    else:
        raise ValueError("unknown covariance expected one of [ewma, moving]")

    return \
        cov_func(df, cov_arg, convert_to) \
        .groupby(level=0) \
        .apply(lambda x: x if x.isnull().values.any() else \
                         _pd.DataFrame(graphical_lasso(x.values, rho, **kwargs)[int(inverse)], index=x.index, columns=x.columns))
