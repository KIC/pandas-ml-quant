from typing import Union, List, Dict, Type, Callable

import numpy as np
import pandas as pd

from ..preprocessing.features_labels import FeaturesLabels, Extractor
from ..typing import MlTypes
from ..utils import multi_index_shape, get_pandas_object, unpack_nested_arrays, has_indexed_columns, pd_cumapply


class MLCompatibleValues(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def cumapply(self, func: callable, start_value=None, **kwargs):
        """
        allows to apply custom cumulative functions

        :param func: callable which receives the current cumulative value and whatever is the series/frame and
        returns the new cumulative value
        :param start_value: the initial value of the current cumulative value
        :param kwargs: args passed to pandas apply
        :return: pandas series or frame as if for the regula apply function
        """
        return pd_cumapply(self.df, func, start_value, **kwargs)

    @property
    def values(self) -> Union[List[np.ndarray], np.ndarray]:
        """
        In contrast to pandas.values the ml.values returns a n-dimensional array with respect to MultiIndex and/or
        nested numpy arrays inside of cells

        :return: numpy array with shape of MultiIndex and/or nested arrays from cells
        """
        return self.get_values()

    def get_values(self, split_multi_index_rows=False, squeeze=False, dtype=None):
        # get raw values
        values = unpack_nested_arrays(self.df, split_multi_index_rows, dtype)

        # return in multi level shape if multi index is used
        def reshape_when_multi_index_column(values):
            if has_indexed_columns(self.df) and isinstance(self.df.columns, pd.MultiIndex):
                index_shape = multi_index_shape(self.df.columns)
                try:
                    # try to reshape the nested arrays into the shape of the multi index
                    values = values.reshape((values.shape[0],) + index_shape + values.shape[len(index_shape):])
                except ValueError as ve:
                    # but it might well be that the shapes do not match, then just ignore the index shape
                    if not "cannot reshape array" in str(ve):
                        raise ve

            if squeeze and values.ndim > 2 and values.shape[2] == 1:
                values = values.reshape(values.shape[:-1])

            return values

        # if values is list reshape each array
        return [reshape_when_multi_index_column(v) for v in values] if isinstance(values, List) else \
            reshape_when_multi_index_column(values)

    def extract(self,
                features_and_labels_definition: FeaturesLabels,
                type_mapping: Dict[Type, Callable[[MlTypes.DataFrame], MlTypes.DataFrame]] = None,
                **kwargs) -> Extractor:
        return Extractor(self.df, features_and_labels_definition, type_mapping, **kwargs)

    def __getitem__(self, item: Union[str, list, callable]) -> Union[pd.Series, pd.DataFrame]:
        """
        This is a magic way to access columns in a DataFrame. We can use regex and even lambdas to select and
        calculate columns.

        df.ML[".*Close$"]  # gets all close columns
        df.ML[lambda df: df["Close"] * 2  # get a column and calculates something

        :param item:
        :return:
        """
        return get_pandas_object(self.df, item)

