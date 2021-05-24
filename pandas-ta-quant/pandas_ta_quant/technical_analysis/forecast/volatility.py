from arch import arch_model  # alternatively https://pyflux.readthedocs.io/en/latest/garch.html

import pandas_ta_quant.technical_analysis.normalizer as _norm
from pandas_ml_common import Typing
from pandas_ta_quant._decorators import *
from functools import partial
import scipy.special as scsp
import numpy as np


@for_each_top_level_row
@for_each_column
def ta_garch11(df: Typing.PatchedPandas, period=200, forecast=1, returns='returns'):
    r = getattr(_norm, f'ta_{returns}')(df) if returns is not None else df

    def model(x):
        model = arch_model(x, p=1, q=1, dist='StudentsT', rescale=True)
        res = model.fit(update_freq=0, disp='off', show_warning=False)

        return res.forecast(horizon=forecast, method='analytic').variance.iloc[-1, -1] / (res.scale * res.scale)

    return r\
        .rolling(period)\
        .apply(model)\
        .rename(f"{df.name}_garch11({period})->{forecast}")


# rough stochastic fractional volatility
def ta_rsfv(df: Typing.PatchedDataFrame, period=255*2, lags=30, delta=1, open="Open", high="High", low="Low", close="Close") -> Typing.PatchedPandas:
    from pandas_ta_quant.technical_analysis import ta_gkyz_volatility, ta_vola_hurst
    rv = ta_gkyz_volatility(df, period=1, open=open, high=high, low=low, close=close)
    dfh = ta_vola_hurst(df, period=period, lags=lags)

    def fc(rv, h, nu, delta=1):
        i = np.arange(len(rv))
        cf = 1. / ((i + 1. / 2.) ** (h + 1. / 2.) * (i + 1. / 2. + delta))
        cf = np.fliplr([cf])[0]

        ldata = np.log(rv)
        fcst = (ldata * cf).sum() / sum(cf)

        def c_tilde(h):
            return scsp.gamma(3. / 2. - h) / scsp.gamma(h + 1. / 2.) * scsp.gamma(2. - 2. * h)

        return np.exp(fcst + 2 * nu ** 2 * c_tilde(h) * delta ** (2 * h))

    partials = dfh.apply(lambda r: partial(fc, h=r[0], nu=r[1], delta=delta), axis=1)
    return rv.rolling(lags).apply(lambda r: partials[r.index[-1]](r)).rename(f"RSFV_{period}_{lags}_{delta}")


