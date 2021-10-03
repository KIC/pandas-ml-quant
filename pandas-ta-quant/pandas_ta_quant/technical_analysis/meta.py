from typing import Any, Callable, Iterable

import pandas as pd

from pandas_ml_common import MlTypes


def ta_repeat(
        df: MlTypes.PatchedPandas,
        func: Callable[[MlTypes.PatchedPandas, Any], MlTypes.PatchedPandas],
        repetition: Iterable,
        multiindex=None,
        *args,
        **kwargs):
    res = pd.concat([func(df, param, *args, **kwargs) for param in repetition], axis=1)

    if multiindex is not None:
        res.columns = pd.MultiIndex.from_product([[multiindex], res.columns])

    return res
