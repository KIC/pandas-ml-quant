import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

from pandas_ml_common import MlTypes
from pandas_ml_common.utils import get_pandas_object, intersection_of_index, unpack_nested_arrays
from pandas_ml_common.utils.normalization import ecdf
from pandas_ta_quant._decorators import *
from pandas_ta_quant.technical_analysis import ta_log_returns


@for_each_top_level_row
@for_each_column
def ta_ncdf_compress(df: MlTypes.PatchedPandas, period=200, lower_percentile=25, upper_percentile=75) -> MlTypes.PatchedPandas:
    f50 = df.rolling(period).mean().rename("f50")
    fup = df.rolling(period).apply(lambda x: np.percentile(x, upper_percentile)).rename("fup")
    flo = df.rolling(period).apply(lambda x: np.percentile(x, lower_percentile)).rename("flo")

    return pd.Series(norm.cdf(0.5 * (df - f50) / (fup - flo)) - 0.5, index=df.index, name=df.name)


@for_each_top_level_row
@for_each_column
def ta_z_norm(df: MlTypes.PatchedPandas, period=200, ddof=1, demean=True, lag=0):
    # (value - mean) / std
    s = df.rolling(period).std(ddof=ddof)
    a = (df - df.rolling(period).mean().shift(lag)) if demean else df
    return (a / s / 4).rename(df.name)


@for_each_top_level_row
def ta_sma_price_ratio(df: MlTypes.Series, period=14, log=False):
    from .labels.continuous import ta_future_pct_to_current_mean
    return ta_future_pct_to_current_mean(df, 0, period, log)


@for_each_top_level_row
def _ta_adaptive_normalisation():
    # TODO implement .papers/Adaptive Normalization.pdf
    pass


@for_each_top_level_row
def ta_normalize_row(df: MlTypes.PatchedDataFrame, normalizer: str = "uniform", level=None):
    # normalizer can be one of minmax01, minmax-11, uniform, standard or callable
    if isinstance(df.columns, pd.MultiIndex) and level is not None:
        return for_each_top_level_column(ta_normalize_row, level=level)(df, normalizer)
    else:
        def scaler(row):
            values = unpack_nested_arrays(row, split_multi_index_rows=False)
            values_2d = values.reshape(-1, 1)

            if normalizer == 'minmax01':
                return MinMaxScaler().fit(values_2d).transform(values_2d).reshape(values.shape)
            elif normalizer == 'minmax-11':
                return MinMaxScaler(feature_range=(-1, 1)).fit(values_2d).transform(values_2d).reshape(values.shape)
            elif normalizer == 'standard':
                # (value - mean) / std
                return values - values.mean() / np.std(values)
            elif normalizer == 'uniform':
                return ecdf(values_2d).reshape(values.shape)
            elif callable(normalizer):
                return normalizer(row)
            else:
                raise ValueError('unknown normalizer need to one of: [minmax01, minmax-11, uniform, standard, callable(r)]')

        return df.apply(scaler, axis=1, result_type='broadcast')


@for_each_top_level_row
def ta_delta_hedged_price(df: MlTypes.PatchedDataFrame, benchmark):
    df_bench = get_pandas_object(df, benchmark)
    idx = intersection_of_index(df, df_bench)

    df = df.loc[idx]
    df_bench = df_bench.loc[idx]

    if hasattr(df, "columns") and not isinstance(benchmark, MlTypes.AnyPandasObject) and benchmark in df.columns:
        df = df.drop(benchmark, axis=1)

    bench_returns = ta_log_returns(df_bench)
    if df.ndim > 1:
        bench_returns = np.repeat(bench_returns.values.reshape(-1, 1), df.shape[1], axis=1)

    delta_hedged = ta_log_returns(df) - bench_returns
    return np.exp(delta_hedged.cumsum())


@for_each_top_level_row
@for_each_top_level_column
def ta_zscored_candle(df: MlTypes.PatchedDataFrame, open="Open", high="High", low="Low", close="Close", volume="Volume", period=20, ddof=1):
    fix_domain_constant = 4

    @for_each_column
    def ta_zscoreing(df: pd.DataFrame) -> pd.DataFrame:
        mean = df.rolling(period).mean().rename("mean")
        std = df.rolling(period).std(ddof=ddof).rename("std")
        return pd.concat([mean, std], axis=1)

    scoring = ta_zscoreing(df[close])

    # trend
    scoring[f"zmean"] = scoring["mean"].pct_change()

    # candle
    for col in [open, high, low, close]:
        scoring[f"z{col}"] = ((df[col] - scoring["mean"]) / scoring["std"]) / fix_domain_constant

    # body of candle
    scoring[f"zbody"] = scoring[f"z{close}"] - scoring[f"z{open}"]

    # upper shadow
    scoring[f"zupper"] = scoring[f"z{high}"] - scoring[[f"z{open}", f"z{close}"]].max(axis=1)

    # lower shadow
    scoring[f"zlower"] = scoring[[f"z{open}", f"z{close}"]].min(axis=1) - scoring[f"z{low}"]

    # scored standard deviation
    scoring["zstd"] = ((scoring["mean"] + scoring["std"]) / (scoring["mean"] - scoring["std"]) - 1)

    if volume is not None:
        vscoring = ta_zscoreing(df[volume])
        scoring["zvol"] = ((df[volume] - vscoring["mean"]) / vscoring["std"]) / fix_domain_constant

    return scoring.drop(["mean", "std"], axis=1)
