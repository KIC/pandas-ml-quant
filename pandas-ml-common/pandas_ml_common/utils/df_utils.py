from typing import List

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from .index_utils import flatten_multi_column_index


#def pd_concat(frames: pd.DataFrame, default=None, *args, **kwargs):
#    return pd.concat(frames, *args, **kwargs) if len(frames) > 0 else default


def pd_dropna(frame: pd.DataFrame):
    if frame is None:
        return None

    if isinstance(frame, list):
        return [pd_dropna(f) for f in frame]

    return frame.dropna()


def pd_concat(frames, multiindex_columns=True, join='outer', **kwargs):
    if frames is None:
        return None

    if not isinstance(frames, List):
        return frames

    if len(frames) <= 0:
        return None

    valid_frames = \
        [flatten_multi_column_index(f, as_string=True, prefix=None if multiindex_columns else i) for i, f in
         enumerate(frames) if f is not None]

    if len(valid_frames) <= 0:
        return None

    return pd.concat(
        valid_frames,
        axis=1,
        join=join,
        names=list(range(len(frames))) if multiindex_columns else None,
        **kwargs
    ) if len(frames) > 1 else frames[0]


def fix_multiindex_row_asymetry(df, default=np.NaN, sort=False):
    return fix_multiindex_asymetry(df, default, False, axis=0, sort=sort)


def fix_multiindex_column_asymetry(df, default=np.NaN, inplace=False, sort=False):
    return fix_multiindex_asymetry(df, default, inplace=inplace, axis=1, sort=sort)


def fix_multiindex_asymetry(df, default=np.NaN, inplace=False, axis=0, sort=False):
    df = df.T.copy() if axis == 0 else df
    idx = df.columns

    if not isinstance(idx, pd.MultiIndex):
        return df

    df = df if inplace else df.copy()
    all_levels_keys = [set(idx.get_level_values(l)) for l in range(idx.nlevels)]

    for k in pd.MultiIndex.from_product(all_levels_keys).to_list():
        if k not in idx:
            df[k] = default

    df = df.T.copy() if axis == 0 else df
    return df.sort_index(axis=axis) if sort else df


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
