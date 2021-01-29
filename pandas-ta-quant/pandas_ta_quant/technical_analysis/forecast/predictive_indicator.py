import warnings
from functools import partial
from typing import Iterable

import numpy as np
import logging
from pandas_ta_quant._decorators import *
from pandas_ml_common import Typing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

_log = logging.getLogger(__name__)


@for_each_top_level_row
@for_each_column
def ta_hmm(df: Typing.PatchedSeries, nr_components, means_estimator, period=250, forecast_period=1, n_iter=500):
    from hmmlearn import hmm

    X = df.to_frame().pct_change(forecast_period).dropna().values

    def moving_hmm(X):
        # new model
        model = hmm.GaussianHMM(n_components=nr_components, n_iter=n_iter, covariance_type="diag", init_params="tc")

        # set means
        model.means_ = np.array(means_estimator(X)).reshape((nr_components, X.shape[1]))

        # set initial state probability
        model.startprob_ = np.zeros(nr_components)
        model.startprob_[np.abs(model.means_ - X[0]).argmin()] = 1.0

        # fit model
        model = model.fit(X)

        # predict current
        hidden_state_probabilities = model.predict_proba(X)
        probabilities = hidden_state_probabilities[-1:]
        transition_matrix = model.transmat_
        next_state_probabilities = probabilities @ transition_matrix

        return next_state_probabilities.reshape(nr_components)

    p = period - 1
    estimates = np.array([moving_hmm(X[i - p:i]) for i in range(p, len(X))])
    print(estimates.shape)
    return pd.DataFrame(estimates, index=df.index[p+forecast_period:])


@for_each_top_level_row
@for_each_column
def ta_sarimax(df: Typing.PatchedSeries, period=60, forecast=1, order=(1, 0, 1), alpha=0.5):
    assert forecast > 0, "forecast need to be > 0"
    forecast = forecast if isinstance(forecast, Iterable) else [forecast]
    res = pd.DataFrame({}, index=df.index)
    dfclean = df.dropna()

    def arima(x, fc):
        try:
            forecasted = SARIMAX(x, order=order).fit(disp=-1).forecast(fc, alpha=alpha)
        except Exception as e:
            _log.warning(f"failed arma model: {e}")
            forecasted = np.nan

        return forecasted  # TODO eventually return conficence as well, conf_int[:, 0], conf_int[:, 1]

    for fc in forecast:
        res = res.join(dfclean.rolling(period).apply(partial(arima, fc=fc), raw=True).rename(f'{df.name}_sarimax_fc_{fc}'))

    return res



