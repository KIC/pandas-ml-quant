from typing import Tuple, Union

import pandas as pd
import tensorflow as tf
from tensorflow import float32

from pandas_ml_common.decorator import MultiFrameDecorator


def from_pandas(df: pd.DataFrame, default: tf.Tensor = None) -> Union[tf.Tensor, Tuple[tf.Tensor, ...]]:
    """
    tf.convert_to_tensor(
        value, dtype=None, dtype_hint=None, name=None
    )
    :param df: the datafrae to convert
    :param default: an optional default value if the dataframe is none
    :return: a tensorflow tensor
    """

    if df is None and default is None:
        return None

    if isinstance(df, MultiFrameDecorator):
        return tuple([from_pandas(f, default) for f in df.frames()])

    return tf.convert_to_tensor(
        df._.get_values(split_multi_index_rows=False, squeeze=True),
        dtype=float32,
        name=f"df({id(df)})"
    ) if df is not None else default

