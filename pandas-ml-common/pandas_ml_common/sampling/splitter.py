import logging
from typing import Tuple, Callable

import pandas as pd

_log = logging.getLogger(__name__)


def random_splitter(test_size=0.3, youngest_size: float = None, seed=42) -> Callable[[pd.Index], Tuple[pd.Index, pd.Index]]:
    from sklearn.model_selection import train_test_split

    youngest_split_size = test_size * youngest_size if youngest_size is not None else 0
    random_sample_test_size = test_size * (1 - youngest_split_size)

    def splitter(index: pd.Index) -> Tuple[pd.Index, pd.Index]:
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

    return splitter


def naive_splitter(test_size=0.3) -> Callable[[pd.Index], Tuple[pd.Index, pd.Index]]:
    def splitter(index: pd.Index) -> Tuple[pd.Index, pd.Index]:
        end_idx = int(len(index) * (1 - test_size))
        return index[0:end_idx], index[end_idx:]

    return splitter


dummy_splitter = naive_splitter(0)