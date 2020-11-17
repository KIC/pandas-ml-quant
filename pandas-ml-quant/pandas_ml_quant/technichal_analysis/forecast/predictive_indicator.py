import numpy as np

from pandas_ml_quant.technichal_analysis._decorators import *
from pandas_ml_utils import Typing


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

