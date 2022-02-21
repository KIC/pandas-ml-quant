from typing import Any, Callable, Iterable, Dict, Union

import pandas as pd

from pandas_ml_common import MlTypes
from pandas_ta_quant._decorators import for_each_top_level_row, for_each_top_level_column


def ta_repeat(
        df: MlTypes.PatchedPandas,
        func: Callable[[MlTypes.PatchedPandas, Any], MlTypes.PatchedPandas],
        repetition: Iterable,
        multiindex=None,
        *args,
        **kwargs):
    res = pd.concat([pd.DataFrame({}, index=df.index)] + [func(df, param, *args, **kwargs) for param in repetition], axis=1, join='outer')

    if multiindex is not None:
        res.columns = pd.MultiIndex.from_product([[multiindex], res.columns])

    return res


@for_each_top_level_row
@for_each_top_level_column
def ta_apply(df: MlTypes.PatchedPandas, func: Union[Callable, Dict[str, Callable]], period=None, columns=None):
    if isinstance(func, dict):
        keys = []
        frames = []
        for k, f in func.items():
            frames.append(ta_apply(df, f, period=period, columns=columns))
            keys.append(k)

        res = pd.concat(frames, axis=1, names=keys)

        if not isinstance(res.columns, pd.MultiIndex):
            res.columns = keys

        return res

    if columns:
        df = df[columns]

    def as_pandas(x):
        if not isinstance(x, (pd.Series, pd.DataFrame)):
            if hasattr(x, 'ndim'):
                if x.ndim > 1:
                    return pd.DataFrame(x)

            return pd.DataFrame(x if isinstance(x, (list, set, tuple)) else [x]).T

    if not period:
        return df.apply(func, axis=1, result_type='reduce')
    else:
        rdf = pd.concat(
            [as_pandas(func(df.iloc[i - period: i])) for i in range(period, len(df))],
            axis=0
        )

        rdf.index = df.index[period:]
        return pd.concat([pd.DataFrame({}, index=df.index), rdf], axis=1, join='outer')


@for_each_top_level_row
def ta_resample(df: MlTypes.PatchedPandas, func, freq='D', **kwargs):
    return df.resample(freq, **kwargs).apply(func)
