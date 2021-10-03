import datetime
import logging
from contextlib import redirect_stdout
from functools import partial
from typing import List, Iterable

import numpy as np
import pandas as pd

from pandas_ml_common.utils import fix_multiindex_row_asymetry

_log = logging.getLogger(__name__)


def plot_surface(df_chain: pd.DataFrame, put_columns: List, call_columns: List, risk_free_rate: float = 0, spot: float = None):
    df_greeks = calc_greeks(df_chain, put_columns, call_columns, risk_free_rate, spot)

    #iv = fix_multiindex_column_asymetry(trimmed_chain[["put_IV"]].T).sort_index(axis=1)
    #x, y, z = np.array(list(set(trimmed_chain.index.get_level_values(1)))), np.array(
    #    list(set(trimmed_chain.index.get_level_values(0)))), iv.ML.values[0]
    #y = np.linspace(0, 1, len(y))
    #
    #fig = plt.figure(figsize=(20, 10))
    #
    ## add the subplot with projection argument
    #ax = fig.add_subplot(111, projection='3d')
    #
    ## set labels
    #ax.set_xlabel('Strike price')
    #ax.set_ylabel('Days to expiration')
    #ax.set_zlabel('IV')
    #ax.set_title('Surface')
    #
    ## plot
    #ax.plot_surface(X, Y, z, color='b')


def calc_greeks(
        df_chain: pd.DataFrame,
        put_columns: List,
        call_columns: List,
        risk_free_rate: float = 0,
        spot: float = None,
        date: str = None,
        greeks: Iterable[str] = ("delta", "gamma", "vega", "theta", "rho", "vanna", "charm"),
        cut_tails_after: float = None,
        interpolate_missing: bool = False):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
    from py_lets_be_rational.exceptions import BelowIntrinsicException
    from scipy.stats import norm

    if cut_tails_after is not None:
        df_chain = df_chain[df_chain["dist_pct_spot"].abs() <= cut_tails_after]

    df = pd.DataFrame(index=df_chain.index)
    as_of_date = pd.Timestamp.today(tz=df.index[0][0].tz) if date is None else pd.to_datetime(date)

    if not isinstance(put_columns, (list, tuple)):
        put_columns = [put_columns]

    if not isinstance(call_columns, (list, tuple)):
        call_columns = [call_columns]

    def iv(r, column, type):
        p = r[column]
        duration = _duration(r.name[0], as_of_date)
        s = r.name[1] / (r['dist_pct_spot'] + 1) if spot is None else spot

        try:
            return implied_volatility(p, s, r.name[1], duration, risk_free_rate, type)
        except BelowIntrinsicException:
            return np.NaN
        except Exception as e:
            _log.error(f"iv(price={p}, spot={s}, strike={r.name[1]}, duration={duration}, rate={risk_free_rate}, type={type})")
            raise e

    def interpolate_missing_iv_per_expiration(iv):
        iv = iv.replace(0, None)
        for exp in set(iv.index.get_level_values(0)):
            strike = iv[exp]

            max_otm_put, max_otm_call = strike.first_valid_index(), strike.last_valid_index()
            if max_otm_put is None or max_otm_call is None: continue

            strike = strike[max_otm_put:max_otm_call+0.00001]
            nans = strike.isna()
            xy = strike[~nans]

            if len(xy) <= 4:
                continue

            x = xy.index.values
            y = xy.values
            coeff = np.polyfit(x, y, 2)
            strike[nans] = np.clip(np.polyval(coeff, strike[nans].index.values), 0, None)
            iv[exp][strike.index] = strike.values

        return iv

    def interpolate_missing_iv_per_strike(iv):
        iv = iv.replace(0, None)
        for strike in set(iv.index.get_level_values(1)):
            expirations = iv[pd.IndexSlice[:, strike, :]]
            expirations = expirations[expirations.first_valid_index():expirations.last_valid_index() + pd.Timedelta(seconds=1)]
            nans = expirations.isna()
            xy = expirations[~nans]

            if len(xy) <= 4:
                continue

            x = xy.index.values.astype(np.int64) // 10 ** 9
            y = xy.values
            coeff = np.polyfit(x, y, 2)

            expirations[nans] = np.clip(np.polyval(coeff, expirations[nans].index.values.astype(np.int64) // 10 ** 9), 0, None)
            for exp, v in expirations.iteritems():
                iv[(exp, strike)] = v

        return iv

    def bs_greeks(r, column, type):
        duration = _duration(r.name[0], as_of_date)
        days_in_year = 366.0 if pd.Timestamp.today().is_leap_year else 365.0
        s = r.name[1] / (r['dist_pct_spot'] + 1) if spot is None else spot
        k = r.name[1]
        sigma = r[f"{column}_iv"]

        sigmaT = sigma * duration ** 0.5
        d1 = (np.log(s / k) + (risk_free_rate + 0.5 * (sigma ** 2)) * duration) / sigmaT
        d2 = d1 - sigmaT

        res = {}
        if "delta" in greeks:
            res[f"{c}_delta"] = delta(type, s, k, duration, risk_free_rate, sigma)
        if "gamma" in greeks:
            res[f"{c}_gamma"] = gamma(type, s, k, duration, risk_free_rate, sigma)
        if "vega" in greeks:
            res[f"{c}_vega"] = vega(type, s, k, duration, risk_free_rate, sigma)
        if "theta" in greeks:
            res[f"{c}_theta"] = theta(type, s, k, duration, risk_free_rate, sigma)
        if "rho" in greeks:
            res[f"{c}_rho"] = rho(type, s, k, duration, risk_free_rate, sigma)
        if "vanna" in greeks:
            # sensitivity of delta with respect to 1% volatility
            res[f"{c}_vanna"] = 0.01 * -np.e ** duration * d2 / sigma * norm.pdf(d1)
        if "charm" in greeks:
            # sensitivity of delta with respect to time
            try:
                res[f"{c}_charm"] = (1.0 / days_in_year) * -np.e ** duration * (norm.pdf(d1) * (risk_free_rate / sigmaT - d2 / (2 * duration)))
            except ZeroDivisionError as zde:
                _log.warning(f"zero division for charm(t={duration} sigma_d2={sigmaT - d2}")
                res[f"{c}_charm"] = np.NaN

        return res

    # something prints empty lines, wrap this shit
    with redirect_stdout(None):
        df["time_to_expiration"] =\
            df_chain.index.get_level_values(0).to_series().apply(partial(_duration, today=as_of_date, in_years=False)).values

        for tpe, columns in [('p', put_columns), ('c', call_columns)]:
            for c in columns:
                # calc IV
                df[f"{c}_iv"] = df_chain.apply(partial(iv, column=c, type=tpe), axis=1)

                # interpolate missing values
                if interpolate_missing:
                    df[f"{c}_iv"] = df[[f"{c}_iv"]].apply(interpolate_missing_iv_per_expiration, axis=0)
                    df[f"{c}_iv"] = df[[f"{c}_iv"]].apply(interpolate_missing_iv_per_strike, axis=0)

                # add greeks: Delta, Gamma, Vega, Theta
                df = df.join(df_chain.join(df[f"{c}_iv"]).apply(partial(bs_greeks, column=c, type=tpe), axis=1, result_type='expand'))

            # add center spot distance
            if tpe == 'p':
                df["dist_pct_spot"] = df_chain["dist_pct_spot"] if spot is None else df.index.get_level_values(1) / spot - 1

        df = df.sort_index(axis=0)

        # TODO monkey patch data frame for surface plotting
        # setattr(df, "calculate_greeks", property(lambda f, **kwargs: calc_greeks(f, *symbol_implementation.put_columns_call_columns())))
        return df


def get_otm_only(df: pd.DataFrame, put_prefix='put_', call_prefix='call_', force_symmetric=True):
    put_columns = [c for c in df.columns.tolist() if c.startswith(put_prefix)]
    call_columns = [c for c in df.columns.tolist() if c.startswith(call_prefix)]
    other_columns = [c for c in df.columns.tolist() if not c.startswith(put_prefix) and not c.startswith(call_prefix)]

    res = pd.concat(
        [
            df[df["dist_pct_spot"] < 0][put_columns].rename(columns=lambda s: s.replace(put_prefix, "")),
            df[df["dist_pct_spot"] >= 0][call_columns].rename(columns=lambda s: s.replace(call_prefix, ""))
        ],
        axis=0,
        sort=True
    )

    return (fix_multiindex_row_asymetry(res, sort=True) if force_symmetric else res).join(df[other_columns], how='inner')


def _duration(t: pd.Timestamp, today: pd.Timestamp, in_years=True):
    # we need to add one expiration day to the option to simulate an EndOfDay expiration !
    tdelta = (t + pd.Timedelta(days=1) - today)
    days = sum([365 if y % 4 != 0 or (y % 100 == 0 and y % 400 != 0) else 366 for y in range(t.year, today.year + 1)])

    dur = (tdelta.days + tdelta.seconds / (60. * 60. * 24.)) / (float(days) if in_years else 1)
    if dur < 0:
        _log.warning(f"negative duration {dur} for {t} - {today}, clipped to 1e-9")

    return max(1e-9, dur)
