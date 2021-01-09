from arch import arch_model  # alternatively https://pyflux.readthedocs.io/en/latest/garch.html

import pandas_ta_quant.technical_analysis.normalizer as _norm
from pandas_ml_common import Typing
from pandas_ta_quant._decorators import *


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

