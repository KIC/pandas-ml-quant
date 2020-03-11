from typing import Union

import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def ta_one_hot_encode_discrete(po: Union[pd.Series, pd.DataFrame], drop_na=True) -> Union[pd.Series, pd.DataFrame]:
    if hasattr(po, "columns"):
        return pd.DataFrame([ta_one_hot_encode_discrete(po[col]) for col in po.columns]).T
    else:
        if drop_na:
            po = po.dropna()

        values = po.values.astype(int)
        offset = po.min()
        values = values - offset
        nr_of_classes = values.max() + 1

        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(int(nr_of_classes)))
        return pd.Series(label_binarizer.transform(values).tolist(), index=po.index, name=po.name)


