import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

from pandas_ml_common import pd

_log = logging.getLogger(__name__)


def train_test_split(index: pd.Index,
                     test_size: float = 0.4,
                     youngest_size: float = None,
                     seed: int = 42) -> Tuple[pd.Index, pd.Index]:

    # convert data frame index to numpy array
    index = index.values

    if test_size <= 0:
        train, test = index, index[:0]
    elif seed == 'youngest':
        i = int(len(index) - len(index) * test_size)
        train, test = index[:i], index[i:]
    else:
        random_sample_test_size = test_size if youngest_size is None else test_size * (1 - youngest_size)
        random_sample_train_index_size = int(len(index) - len(index) * (test_size - random_sample_test_size))

        if random_sample_train_index_size < len(index):
            _log.warning(f"keeping youngest {len(index) - random_sample_train_index_size} elements in test set")

            # cut the youngest data and use residual to randomize train/test data
            index_train, index_test = \
                sk_train_test_split(index[:random_sample_train_index_size],
                                 test_size=random_sample_test_size, random_state=seed)

            # then concatenate (add back) the youngest data to the random test data
            index_test = np.hstack([index_test, index[random_sample_train_index_size:]])  # index is 1D

            train, test = index_train, index_test
        else:
            train, test = sk_train_test_split(index, test_size=random_sample_test_size, random_state=seed)

    return pd.Index(train), pd.Index(test)

