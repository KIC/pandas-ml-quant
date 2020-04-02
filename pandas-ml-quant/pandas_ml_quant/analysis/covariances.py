# create convenient type hint
import numpy as _np

from pandas_ml_common import Typing


def ta_ewma_covariance(df: Typing.PatchedPandas, convert_to='returns', alpha=0.97):
    data = df.copy()

    if convert_to == 'returns':
        data = df.pct_change()
    if convert_to == 'log-returns':
        data = _np.log(df) - _np.log(df.shift(1))

    data.columns = data.columns.to_list()
    return data.ewm(com=alpha).cov()

