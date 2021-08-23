from typing import Union as _Union

# create convenient type hint
import numpy as _np
import numpy as np
import pandas as _pd
import pandas as pd
from scipy.stats import zscore

import pandas_ta_quant.technical_analysis.bands as _bands
from pandas_ml_common import Typing
from pandas_ml_common.utils import cumcount
from pandas_ta_quant._decorators import *
from pandas_ta_quant.technical_analysis.filters import ta_ema as _ema, ta_wilders as _wilders, ta_sma as _sma
from pandas_ta_quant._utils import with_column_suffix as _wcs, rolling_apply
from scipy.stats import linregress

_PANDAS = _Union[_pd.DataFrame, _pd.Series]


@for_each_top_level_row
def ta_mean_returns(df: Typing.PatchedPandas, period=20) -> _PANDAS:
    return _wcs(f"mean_return_{period}", df.pct_change().rolling(period).mean())


@for_each_column
@for_each_top_level_row
def ta_macd(df: Typing.PatchedPandas, fast_period=12, slow_period=26, signal_period=9, relative=True) -> _PANDAS:
    fast = _ema(df, fast_period)
    slow = _ema(df, slow_period)

    macd = (fast / slow - 1) if relative else (fast - slow)
    signal = _ema(macd, signal_period)
    hist = macd - signal
    suffix = f'{fast_period},{slow_period},{signal_period}'

    for label, frame in {f"macd_{suffix}": macd, f"signal_{suffix}": signal, f"histogram_{suffix}": hist}.items():
        frame.name = label

    macd = macd.to_frame() if isinstance(macd, _pd.Series) else macd
    return macd.join(signal).join(hist)


@for_each_top_level_row
def ta_mom(df: _PANDAS, period=12, relative=True) -> _PANDAS:
    return _wcs(f"mom_{period}", df.pct_change(period) if relative else df.diff(period))


@for_each_top_level_row
def ta_roc(df: _PANDAS, period=12) -> _PANDAS:
    return _wcs(f"roc_{period}", df.pct_change(period))


@for_each_top_level_row
def ta_stddev(df: _PANDAS, period=12, nbdev=1, ddof=1, downscale=True) -> _PANDAS:
    return _wcs(f"stddev_{period}", (df.rolling(period).std(ddof=ddof) * nbdev) / (100 if downscale else 1))


@for_each_top_level_row
def ta_rsi(df: _PANDAS, period=12):
    returns = df.diff()

    pos = _wilders(returns.clip(lower=0), period)
    neg = _wilders(_np.abs(returns.clip(upper=0)), period)

    return _wcs(f"rsi_{period}", pos / (pos + neg), returns)


@for_each_top_level_row
def ta_apo(df: _PANDAS, fast_period=12, slow_period=26, exponential=False, relative=True) -> _PANDAS:
    fast = _ema(df, fast_period) if exponential else _sma(df, fast_period)
    slow = _ema(df, slow_period) if exponential else _sma(df, slow_period)
    apo = (fast / slow) if relative else (fast - slow)
    return _wcs(f"apo_{fast_period},{slow_period},{int(exponential)}", apo, df)


@for_each_top_level_row
def ta_trix(df: _PANDAS, period=30) -> _PANDAS:
    return _wcs(f"trix_{period}", _ema(_ema(_ema(df, period), period), period).pct_change() * 100, df)


@for_each_top_level_row
def ta_ppo(df: _pd.DataFrame, fast_period=12, slow_period=26, exponential=True) -> _PANDAS:
    fast = _ema(df, period=fast_period) if exponential else _sma(df, period=fast_period)
    slow = _ema(df, period=slow_period) if exponential else _sma(df, period=slow_period)
    return _wcs(f"ppo_{fast_period},{slow_period},{int(exponential)}", (fast - slow) / slow, df)


@for_each_top_level_row
def ta_zscore(df: _PANDAS, period=20, ddof=1, downscale=True):
    res = df.rolling(period).apply(lambda c: zscore(c, ddof=ddof)[-1] / (4 if downscale else 1))

    return _wcs(f"z_{period}", res)


@for_each_top_level_row
@for_each_column
def ta_up_down_volatility_ratio(df: Typing.PatchedSeries, period=60, normalize=True, setof_date=False):
    returns = df.pct_change() if normalize else df
    std = _pd.DataFrame({
        "std+": returns[returns > 0].rolling(period).std(),
        "std-": returns[returns < 0].rolling(period).std()
    }, index=returns.index).fillna(method="ffill")

    ratio = (std["std+"] / std["std-"] - 1).rename("std +/-")

    # eventually we can off set the date such that we can fake one continuous data frame
    if setof_date:
        # +7 -1 binds us approximately to the same week day
        ratio.index = ratio.index - _pd.DateOffset(days=(ratio.index[-1] - ratio.index[0]).days + 7 - 1)

    return ratio


@for_each_top_level_row
@for_each_column
def ta_poly_coeff(df: _PANDAS, period=60, degree=2):
    from pandas_ml_common.utils import ReScaler
    from pandas_ml_common import np_nans

    x = _np.linspace(0, 1, period)
    v = df.values
    res = np_nans((len(df), degree + 1))

    for i in range(period, len(df)):
        y =  v[i - period:i]
        rescaler = ReScaler((y.min(), y.max()), (-1, 1))
        p, err, _, _, _ = _np.polyfit(x, rescaler(y), degree, full=True)
        res[i] = _np.hstack([p[:-1], err / period])

    return _pd.DataFrame(res, index=df.index, columns=[*[f'b{b}' for b in range(0, degree)], "mse"])


@for_each_top_level_row
def ta_sharpe_ratio(df: _PANDAS, period=60, ddof=1, risk_free=0, is_returns=False):
    returns = df if is_returns else df.pct_change()
    mean_returns = returns.rolling(period).mean()
    sigma = returns.rolling(period).std(ddof=ddof)
    return (mean_returns - risk_free) / sigma


@for_each_top_level_row
def ta_sortino_ratio(df: _PANDAS, period=60, ddof=1, risk_free=0, is_returns=False):
    returns = df if is_returns else df.pct_change()
    mean_returns = returns.rolling(period).mean()
    sigma = returns.clip(upper=risk_free).rolling(period).std(ddof=ddof)
    return (mean_returns - risk_free) / sigma


@for_each_top_level_row
@for_each_column
def ta_draw_down(s: _PANDAS, return_dates=False, return_duration=False):
    df = s.copy() if s.ndim > 1 else s.to_frame()

    max_peaks_idx = s.expanding(min_periods=1).apply(lambda x: x.argmax()).fillna(0).astype(int).rename("iloc")
    duration_per_peak_iloc = cumcount(max_peaks_idx.to_frame().reset_index()["iloc"]).rename("duration").to_frame()
    duration_per_peak_iloc["iloc_start"] = duration_per_peak_iloc.index.to_series().shift(1).fillna(0).astype(int)
    duration_per_peak_iloc = duration_per_peak_iloc[duration_per_peak_iloc["duration"] > 1]
    df['max_peaks_idx'] = max_peaks_idx

    nw_peaks = pd.Series(s.iloc[max_peaks_idx.values].values, index=s.index)

    df['dd'] = ((s - nw_peaks) / nw_peaks)
    df['mdd'] = df.groupby('max_peaks_idx').dd.apply(lambda x: x.expanding(min_periods=1).apply(lambda y: y.min())).fillna(0)
    df = df.drop('max_peaks_idx', axis=1)

    if return_dates:
        df = df.join(
            pd.DataFrame(
                {
                    "dd_start" : s.index[duration_per_peak_iloc["iloc_start"]],
                    "dd_end" : s.index[duration_per_peak_iloc.index],
                },
                index=s.index[duration_per_peak_iloc.index]
            )
        )

    if return_duration:
        df = df.join(
            pd.Series(
                duration_per_peak_iloc["duration"].values,
                index=s.index[duration_per_peak_iloc.index],
                name="duration"
            )
        )

    return df.drop(s.columns[1] if s.ndim > 1 else s.name, axis=1)


@for_each_top_level_row
def ta_ma_decompose(df: Typing.PatchedPandas, period=50, boost_ma=10):
    ma = _sma(df, period=period)
    base = ma.pct_change() * boost_ma
    ratio = df / ma - 1

    if isinstance(ratio, Typing.Series):
        ratio.name = f'{df.name}_ma_ratio'

    return base.to_frame().join(ratio)


@for_each_top_level_row
def ta_ma_ratio(df: Typing.PatchedPandas, period=12, exponential='wilder') -> _PANDAS:
    if exponential is True:
        return (1 / _ema(df)) * df.values - 1
    if exponential == 'wilder':
        return (1 / _wilders(df)) * df.values - 1
    else:
        return (1 / _sma(df)) * df.values - 1


@for_each_top_level_row
def ta_std_ret_bands_indicator(df: _pd.DataFrame, period=12, stddevs=[2.0], ddof=1, lag=1, scale_lag=True, include_mean=True, inculde_width=True) -> _PANDAS:
    columns = ["width"]
    res = _bands.ta_std_ret_bands(df, period, stddevs, ddof, lag, scale_lag, inculde_width=inculde_width, include_mean=include_mean)

    if isinstance(res.columns, _pd.MultiIndex):
        columns = [c for c in res.columns.to_list() if c[-1] in columns]

    return res[columns]


@for_each_top_level_row
def ta_bbands_indicator(df: _PANDAS, period=12, stddev=2.0, ddof=1) -> _PANDAS:
    columns = ["width", "z-score", "quantile"]
    res = _bands.ta_bbands(df, period, stddev, ddof)

    if isinstance(res.columns, _pd.MultiIndex):
        columns = [c for c in res.columns.to_list() if c[-1] in columns]

    return res[columns]


@for_each_top_level_row
@for_each_column
def ta_slope(df: _PANDAS, period=12):
    x = _np.arange(period)
    return _wcs(f"slope_{period}", df.rolling(period).apply(lambda y: linregress(x, y).slope))


@for_each_top_level_row
@for_each_column
def ta_potential_turning_point(
        df: _PANDAS,
        period=120,
        point_threshold=2,
        degrees=(-90, 90),
        angles=30,
        realtive=True,
        rho_digits=2,
        rescale_digits=4,
        edge_detector='naive',
        **kwargs):
    from pandas_ta_quant.technical_analysis.edge_detect import EDGE_DETECTOR
    from pandas_ta_quant.technical_analysis.normalizer import ta_rescale
    kwargs = {k.replace("edge_", ""): v for k, v in kwargs.items()}
    edge_or_not = EDGE_DETECTOR[edge_detector](df, **kwargs)

    def calc_edge_count(df_edge_or_not):
        df = df_edge_or_not.iloc[:, 0]
        edge_or_not = df_edge_or_not.iloc[:, 1].values

        # set up spaces
        x = _np.linspace(0, 1, len(df))
        y = ta_rescale(df, (0, 1), digits=rescale_digits)

        # select only edge points pus the last point as we want to do like this last point is an edge point
        mask = edge_or_not != 0
        if mask.sum() <= 3: return 0
        mask[-1] = True

        edge_x, edge_y = x[mask], y[mask]
        thetas = _np.deg2rad(_np.linspace(*degrees, len(edge_x) if angles is None else angles))

        # pre compute angeles, calculate rho's -> 2d Matrix [angles, edge_points]
        cos_theta, sin_theta = _np.cos(thetas), _np.sin(thetas)
        rhos = _np.around(_np.outer(cos_theta, edge_x) + _np.outer(sin_theta, edge_y), rho_digits)

        # get the counts of all unique rhos of the last index and return the count
        rhos_of_interest = _np.unique(rhos[:, -1])
        counts = {roi: (rhos[:, :-1] == roi).sum() for roi in rhos_of_interest}

        return sum([v for v in counts.values() if v > point_threshold]) / ((period * len(thetas)) if realtive else 1)

    return rolling_apply(pd.concat([df, edge_or_not], axis=1), period, calc_edge_count, names=f"PPL_{period}")


@for_each_column
def ta_strike(df: _PANDAS, mode: str = None, distance=False) -> _PANDAS:
    """
    calculate the distance to the lower/upper option strike follwoing the rules described at cboe
      -> https://www.cboe.com/exchange_traded_stock/equity_options_spec/

    * 2 1/2 points when the strike price is between $5 and $25
    * 5 points when the strike price is between $25 and $200
    * 10 points when the strike price is over $200

    :param df: the dataframe
    :param mode: None|ceil|floor|both strike (distance) where None means the closest strike whether its up or down
    :param distance: distance in % or absolute number
    :return: distance to strike
    """

    if mode == 'both':
        return pd.concat([ta_strike(df, 'floor', distance), ta_strike(df, 'ceil', distance)], axis=1)

    def rround(x, base):
        r = base * round(x / base)
        if mode == 'floor':
            if x - r < 0:
                r -= base
        elif mode == 'ceil':
            if x - r > 0:
                r += base

        return r

    def strike(nr):
        if nr < 25:
            # round 2.5
            dist = rround(nr, 2.5)
        elif nr < 200:
            # round 5 points
            dist = rround(nr, 5)
        else:
            # round 10 points
            dist = rround(nr, 10)

        return nr / max(dist, 1) - 1 if distance else dist

    return _wcs(f"strike{'_' + mode if mode else ''}{'_dist' if distance else ''}", df.apply(strike))
