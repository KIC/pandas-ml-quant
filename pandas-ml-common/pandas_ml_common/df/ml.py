from typing import Union

import numpy as np
import pandas as pd

from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args
from pandas_ml_common.utils import multi_index_shape, get_pandas_object, unpack_nested_arrays


class ML(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def values(self) -> np.ndarray:
        """
        In contrast to pandas.values the ml.values returns a n-dimensional array with respect to MultiIndex and/or
        nested numpy arrays inside of cells

        :return: numpy array with shape of MultiIndex and/or nested arrays from cells
        """

        # get raw values
        values = unpack_nested_arrays(self.df)

        # return in multi level shape if multi index is used
        if hasattr(self.df, 'columns') and isinstance(self.df.columns, pd.MultiIndex):
            index_shape = multi_index_shape(self.df.columns)
            values = values.reshape((values.shape[0],) + index_shape + values.shape[len(index_shape):])

        return values

    def extract(self, func: callable, *args, **kwargs):
        return call_callable_dynamic_args(func, self.df, *args, **kwargs)

    def __getitem__(self, item: Union[str, list, callable]) -> Union[pd.Series, pd.DataFrame]:
        """
        # FIXME add text, can be regex ... dynamic call etc .. .

        :param item:
        :return:
        """
        return get_pandas_object(self.df, item)

