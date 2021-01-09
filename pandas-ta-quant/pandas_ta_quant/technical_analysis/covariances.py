# create convenient type hint
import numpy as _np
import pandas as _pd

from pandas_ml_common import Typing
from pandas_ta_quant._decorators import for_each_top_level_row


def _normalize(df, convert_to):
    data = df.copy()

    if convert_to == 'returns':
        data = df.pct_change()
    if convert_to == 'log-returns':
        data = _np.log(df) - _np.log(df.shift(1))

    return data


@for_each_top_level_row
def ta_ewma_covariance(df: Typing.PatchedPandas, alpha=0.97, convert_to='returns', **kwargs):
    data = _normalize(df, convert_to)
    data.columns = data.columns.to_list()
    return data.ewm(com=alpha).cov()


@for_each_top_level_row
def ta_moving_covariance(df: Typing.PatchedPandas, period=30, convert_to='returns', **kwargs):
    data = _normalize(df, convert_to)
    data.columns = data.columns.to_list()
    return data.rolling(period).cov()


@for_each_top_level_row
def ta_mgarch_covariance(df: Typing.PatchedPandas, period=30, convert_to='returns', forecast=1, dist='t', **kwargs):
    assert period >= df.shape[1], f"period need to be > {df.shape[1]}"
    data = _normalize(df, convert_to)
    data.columns = data.columns.to_list()

    def mgarch_cov(window):
        import mgarch
        vol = mgarch.mgarch(dist)
        vol.fit(window.values)
        return _pd.DataFrame(
            vol.predict(forecast)["cov"],
            index=_pd.MultiIndex.from_product([[window.index[-1]], window.columns.to_list()]),
            columns=window.columns.to_list()
        )

    return _pd.concat([mgarch_cov(data.iloc[i - period: i]) for i in range(period, len(data) + 1)], axis=0)


@for_each_top_level_row
def ta_sparse_covariance(df: Typing.PatchedPandas, convert_to='returns', covariance='ewma', cov_arg=0.97, rho=0.1, inverse=False, **kwargs):
    from sklearn.covariance import graphical_lasso

    if covariance in ['ewma', 'weighted']:
        cov_func = ta_ewma_covariance
    elif covariance in ['rolling', 'moving']:
        cov_func = ta_moving_covariance
    elif covariance in ['garch', 'mgarch']:
        cov_func = ta_mgarch_covariance
    else:
        raise ValueError("unknown covariance expected one of [ewma, moving]")

    return \
        cov_func(df, cov_arg, convert_to) \
        .groupby(level=0) \
        .apply(lambda x: x if x.isnull().values.any() else \
                         _pd.DataFrame(graphical_lasso(x.values, rho, **kwargs)[int(inverse)], index=x.index, columns=x.columns))





