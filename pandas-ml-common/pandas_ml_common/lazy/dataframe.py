import uuid
from functools import lru_cache
from typing import Callable, Union

import pandas as pd


class LazyDataFrame(pd.DataFrame):

    def __init__(self, df: pd.DataFrame, **kwargs: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]) -> None:
        super().__init__(df)
        self.hash = uuid.uuid4()
        self.df = df
        self.kwargs = kwargs

    @property
    def columns(self):
        return self.df.columns.tolist() + list(self.kwargs.keys())

    def __getitem__(self, item: str):
        return self.to_dataframe()[item]

    def __setitem__(self, key: str, value: Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]):
        self.hash = uuid.uuid4()
        if callable(value):
            self.kwargs[key] = value(self.df)
        else:
            self.df[key] = value

    def __hash__(self):
        return int(self.hash)

    def __eq__(self, other):
        return self.hash == other.hash if isinstance(other, LazyDataFrame) else False

    def __deepcopy__(self, memodict={}):
        return LazyDataFrame(self.df, **self.kwargs)

    def with_dataframe(self, df: pd.DataFrame):
        return LazyDataFrame(df, **self.kwargs)

    @lru_cache(maxsize=1)
    def to_dataframe(self) -> pd.DataFrame:
        df = self.df.copy()
        for key, calculation in self.kwargs.items():
            column = calculation(df)
            if isinstance(column, pd.DataFrame):
                df = df.join(column.add_prefix(f'{key}_')) # TODO multi index ???
            else:
                df[key] = column

        return df