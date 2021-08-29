import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object


def pd_concat(frames: pd.DataFrame, default=None, *args, **kwargs):
    return pd.concat(frames, *args, **kwargs) if len(frames) > 0 else default


def pd_hash(df, columns=None, categorize: bool = False, **kwargs):
    hash = hash_pandas_object(df[columns] if columns is not None else df, categorize=categorize, **kwargs)
    if isinstance(hash, pd.Series):
        hash = hash.sum() * 31

    return hash


def pd_equals(df1, df2, lazy=False, columns=None, decimal=6):
    try:
        df1 = df1[columns] if columns is not None else df1
        df2 = df2[columns] if columns is not None else df2
    except KeyError as ke:
        return False

    if pd_hash(df1) != pd_hash(df2):
        return False

    if df1.index.to_list() != df2.index.to_list():
        return False

    if hasattr(df1, 'columns') and df1.columns.to_list() != df2.columns.to_list():
        return False

    if lazy:
        return np.testing.assert_array_almost_equal(df1.iloc[0].values, df2.iloc[0].values, decimal=decimal) \
            and np.testing.assert_array_almost_equal(df1.iloc[-1].values, df2.iloc[-1].values, decimal=decimal)

    # now we need to implement a heuristic if the data is equal
    if hasattr(df1, 'columns'):
        return pd.testing.assert_frame_equal(
            df1, df2,
            check_datetimelike_compat=True,
            check_categorical=False,
            check_like=True
        )
    else:
        return pd.testing.assert_series_equal(
            df1, df2,
            check_datetimelike_compat=True,
            check_categorical=False
        )
