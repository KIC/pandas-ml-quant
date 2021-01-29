import logging
import math
from typing import Tuple, Callable

import pandas as pd
import numpy as np

from pandas_ml_common.utils import temp_seed, unique_level_rows, concat_indices

_log = logging.getLogger(__name__)


def stratified_random_splitter(test_size=0.3, partition_row_multi_index=False, seed=42) -> Callable[[pd.Index], Tuple[pd.Index, pd.Index]]:
    def choose(x):
        arr = np.empty(len(x[0]), dtype=object)
        arr[:] = x[0]
        return np.random.choice(arr, math.ceil(len(x[0]) * test_size)).tolist(),

    def splitter(index: pd.Index, y, *args, **kwargs) -> Tuple[pd.Index, pd.Index]:
        df = y.copy()
        df["index"] = df.index.to_list()  # we want tuples in case of multi index
        indices_per_class = df.groupby(y.columns.to_list()).agg(lambda x: list(x))

        with temp_seed(seed):
            test_idx = concat_indices(
                [i[0] for i in indices_per_class.apply(choose, axis=1, result_type='reduce').to_list()])

        return index.difference(test_idx), test_idx

    return _multi_index_splitter(splitter) if partition_row_multi_index else splitter


def random_splitter(test_size=0.3, youngest_size: float = None, partition_row_multi_index=False, seed=42) -> Callable[[pd.Index], Tuple[pd.Index, pd.Index]]:
    from sklearn.model_selection import train_test_split

    youngest_split_size = test_size * youngest_size if youngest_size is not None else 0
    random_sample_test_size = test_size * (1 - youngest_split_size)

    def splitter(index: pd.Index, *args, **kwargs) -> Tuple[pd.Index, pd.Index]:
        random_sample_train_index_size = int(len(index) - len(index) * (test_size - random_sample_test_size))

        if test_size <= 0:
            return index, index[:0]
        elif random_sample_train_index_size < len(index):
            _log.warning(f"keeping youngest {len(index) - random_sample_train_index_size} elements in test set")

            # cut the youngest data and use residual to randomize train/test data
            index_train, index_test = \
                train_test_split(
                    index[:random_sample_train_index_size],
                    test_size=random_sample_test_size,
                    random_state=seed
                )

            # then concatenate (add back) the youngest data to the random test data
            index_test = index_test.append(index[random_sample_train_index_size:])

            return index_train, index_test
        else:
            return train_test_split(index, test_size=random_sample_test_size, random_state=seed)

    return _multi_index_splitter(splitter) if partition_row_multi_index else splitter


def naive_splitter(test_size=0.3, partition_row_multi_index=False) -> Callable[[pd.Index], Tuple[pd.Index, pd.Index]]:
    def splitter(index: pd.Index, *args, **kwargs) -> Tuple[pd.Index, pd.Index]:
        end_idx = int(len(index) * (1 - test_size))
        return index[0:end_idx], index[end_idx:]

    return _multi_index_splitter(splitter) if partition_row_multi_index else splitter


def duplicate_data() -> Callable[[pd.Index], Tuple[pd.Index, pd.Index]]:
    def splitter(index: pd.Index, *args) -> Tuple[pd.Index, pd.Index]:
        return index, index

    return splitter


# and add some aliases
dummy_splitter = naive_splitter(0)
timeseries_splitter = naive_splitter


# private functions
def _multi_index_splitter(splitter) -> Tuple[pd.Index, pd.Index]:
    def _splitter(index: pd.Index, *args, **kwargs):
        if isinstance(index, pd.MultiIndex):
            group_indexes = [splitter(index[index.get_loc(group)], *args, **kwargs) for group in unique_level_rows(index)]
            return concat_indices([gi[0] for gi in group_indexes]), concat_indices([gi[1] for gi in group_indexes])
        else:
            splitter(index, *args, **kwargs)

    return _splitter

