from contextlib import redirect_stdout
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from pandas_ml_common.utils import fix_multiindex_row_asymetry


def plot_surface(df_chain: pd.DataFrame, put_columns: List, call_columns: List, risk_free_rate: float = 0, spot: float = None):
    df_greeks = calc_greeks(df_chain, put_columns, call_columns, risk_free_rate, spot)

    #iv = fix_multiindex_column_asymetry(trimmed_chain[["put_IV"]].T).sort_index(axis=1)
    #x, y, z = np.array(list(set(trimmed_chain.index.get_level_values(1)))), np.array(
    #    list(set(trimmed_chain.index.get_level_values(0)))), iv._.values[0]
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


def calc_greeks(df_chain: pd.DataFrame, put_columns: List, call_columns: List, risk_free_rate: float = 0, spot: float = None, cut_tails_after: float = None, interpolate_missing: bool = False):
    from py_vollib.black_scholes.implied_volatility import implied_volatility
    from py_vollib.black_scholes.greeks.analytical import delta
    from py_lets_be_rational.exceptions import BelowIntrinsicException

    if cut_tails_after is not None:
        df_chain = df_chain[df_chain["dist_pct_spot"].abs() <= cut_tails_after]

    df = pd.DataFrame(index=df_chain.index)
    today = pd.Timestamp.today(tz=df.index[0][0].tz)

    if not isinstance(put_columns, (list, tuple)):
        put_columns = [put_columns]

    if not isinstance(call_columns, (list, tuple)):
        call_columns = [call_columns]

    def iv(r, column, type):
        duration = _duration(r.name[0], today)
        s = r.name[1] / (r['dist_pct_spot'] + 1) if spot is None else spot

        try:
            return implied_volatility(r[column], s, r.name[1], duration, risk_free_rate, type)
        except BelowIntrinsicException:
            return np.NaN

    def interpolate_missing_iv_per_expiration(iv):
        iv = iv.replace(0, None)
        for exp in set(iv.index.get_level_values(0)):
            strike = iv[exp]
            strike = strike[strike.first_valid_index():strike.last_valid_index()+0.00001]
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

    def d(r, column, type):
        duration = _duration(r.name[0], today)
        s = r.name[1] / (r['dist_pct_spot'] + 1) if spot is None else spot
        return delta(type, s, r.name[1], duration, risk_free_rate, r[f"{column}_iv"])

    # something prints empty lines, wrap this shit
    with redirect_stdout(None):
        df["time_to_expiration"] =\
            df_chain.index.get_level_values(0).to_series().apply(partial(_duration, today=today, in_years=False)).values

        for tpe, columns in [('p', put_columns), ('c', call_columns)]:
            for c in columns:
                # calc IV
                df[f"{c}_iv"] = df_chain.apply(partial(iv, column=c, type=tpe), axis=1)

                # interpolate missing values
                if interpolate_missing:
                    df[f"{c}_iv"] = df[[f"{c}_iv"]].apply(interpolate_missing_iv_per_expiration, axis=0)
                    df[f"{c}_iv"] = df[[f"{c}_iv"]].apply(interpolate_missing_iv_per_strike, axis=0)

                # add greeks
                df[f"{c}_delta"] = df_chain.join(df[f"{c}_iv"]).apply(partial(d, column=c, type=tpe), axis=1)

            # add center spot distance
            if tpe == 'p':
                df["dist_pct_spot"] = df_chain["dist_pct_spot"] if spot is None else df.index.get_level_values(1) / spot - 1

        return df.sort_index(axis=0)


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
    tdelta = (t - today)
    days = sum([365 if y % 4 != 0 or (y % 100 == 0 and y % 400 != 0) else 366 for y in range(t.year, today.year + 1)])

    return (tdelta.days + tdelta.seconds / (60. * 60. * 24.)) / (float(days) if in_years else 1)