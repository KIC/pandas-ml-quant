from typing import Union, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

from pandas_ta_quant._decorators import for_each_column


@for_each_column
def ta_one_hot_encode_discrete(po: Union[pd.Series, pd.DataFrame], drop_na=True, nr_of_classes=None, offset=None, expand=False) -> Union[pd.Series, pd.DataFrame]:
    if drop_na:
        po = po.dropna()

    if offset is None:
        offset = po.min()

    values = po.values.astype(int)
    values = values - offset

    if nr_of_classes is None:
        nr_of_classes = values.max() + 1

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(int(nr_of_classes)))

    if expand:
        columns = expand if isinstance(expand, Iterable) else None
        return pd.DataFrame(label_binarizer.transform(values), index=po.index, columns=columns)
    else:
        return pd.Series(label_binarizer.transform(values).tolist(), index=po.index, name=po.name)


def ta_one_hot_argmax(df: pd.DataFrame):
    argmax = df.apply(np.argmax, axis=1).values
    res = np.zeros(df.shape)
    res[np.arange(argmax.size), argmax] = 1
    resdf = pd.DataFrame(res, index=df.index, columns=df.columns)

    resdf.loc[pd.isna(df).any(1), :] = np.NaN
    return resdf