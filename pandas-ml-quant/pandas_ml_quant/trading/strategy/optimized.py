import logging
import traceback

import numpy as np

from pandas_ml_quant.analysis import ta_ewma_covariance
from pandas_ml_common import pd, Typing
from qpsolvers import solve_qp
from statsmodels.stats.correlation_tools import cov_nearest



_log = logging.getLogger(__name__)


def ta_markowitz(df: Typing.PatchedDataFrame,
                 covariances=None,
                 risk_aversion=5,
                 return_period=60,
                 prices='Close',
                 expected_returns=None,
                 rebalance_trigger=None,
                 solver='cvxopt',
                 tail=None):
    assert isinstance(df.columns, pd.MultiIndex), \
        "expect multi index columns 'prices', 'expected returns' and rebalance trigger"

    # risk
    if covariances is None:
        cov = ta_ewma_covariance(df._[prices])
    elif isinstance(covariances, str):
        cov = df._[covariances]
    else:
        cov = covariances
    cov = cov.dropna()

    # return
    exp_ret = _default_returns_estimator(df, prices, expected_returns, return_period, len(cov.columns))
    exp_ret.columns = cov.columns

    # re-balance
    trigger = (pd.Series(np.ones(len(df)), index=df.index) if rebalance_trigger is None else df[rebalance_trigger]).dropna()

    # non negative weight constraint and weights sum to 1
    h = np.zeros(len(cov.columns)).reshape((-1, 1))
    G = -np.eye(len(h))
    A = np.ones(len(h)).reshape((1, -1))
    b = np.ones(1)

    # magic solution's
    keep_solution = (np.empty(len(h)) * np.nan)
    uninvest = np.zeros(len(h))

    # keep last solution
    last_solution = None

    def optimize(t, sigma, pi):
        nonlocal last_solution
        nr_of_assets = len(sigma)

        # only optimize if we have a re-balance trigger (early exit)
        if last_solution is not None and last_solution.sum() > 0.99:
            # so we had at least one valid solution in the past
            # we can early exit if we do not have any signal or or no signal for any currently hold asset
            if t.ndim > 1 and t.shape[1] == nr_of_assets:
                if t[:, last_solution >= 0.01].sum().any() < 1:
                    return keep_solution
            else:
                if t.sum().any() < 1:
                    return keep_solution

        # make sure covariance matrix is positive definite
        simga = cov_nearest(sigma)

        # we perform optimization except when all expected returns are < 0
        # then we early exit with an un-invest command
        if len(pi[:, pi[0] < 0]) == pi.shape[1]:
            return uninvest
        else:
            try:
                sol = solve_qp(risk_aversion * sigma, -pi.T, G=G, h=h, A=A, b=b, solver=solver)
                if sol is None:
                    _log.error("no solution found")
                    return uninvest
                else:
                    return sol
            except Exception as e:
                _log.error(traceback.format_exc())
                return uninvest

    index = sorted(set(df.index.intersection(cov.index.get_level_values(0)).intersection(exp_ret.index).intersection(trigger.index)))
    if tail is not None:
        index = index[-abs(tail)]

    weights = [optimize(trigger.loc[[i]].values, cov.loc[[i]].values, exp_ret[cov.columns].loc[[i]].values) for i in index]

    # turn weights into a data frame
    return pd.DataFrame(weights, index=index, columns=cov.columns)


def _default_returns_estimator(df, prices, expected_returns, return_period, nr_of_assets):
    # return
    if expected_returns is None:
        exp_ret = df._[prices].pct_change().rolling(return_period).mean()
    elif isinstance(expected_returns, (int, float, np.ndarray)):
        exp_ret = pd.Series(np.ones((len(df), nr_of_assets)) * expected_returns, index=df.index)
    elif isinstance(expected_returns, str):
        exp_ret = df._[expected_returns]
    else:
        exp_ret = expected_returns

    return exp_ret.dropna()
