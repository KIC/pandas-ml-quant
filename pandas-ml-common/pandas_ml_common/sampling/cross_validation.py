from typing import Generator, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import _num_samples

from pandas_ml_common.utils import temp_seed, call_callable_dynamic_args, unique_level_rows, PandasObject
import pandas as pd


class KFoldBoostRareEvents(KFold):

    def __init__(self, n_splits='warn', shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        # all events which are true are considered to be rare
        rare_event_indices = indices[np.sum(y, axis=tuple(range(1, y.ndim))) >= 0.999]

        for f, (train_idx, test_idx) in enumerate(super().split(X, y, groups)):
            yield np.hstack([train_idx, rare_event_indices]), np.hstack([test_idx, rare_event_indices])


class KEquallyWeightEvents(KFold):

    def __init__(self, n_splits='warn', seed=None):
        super().__init__()
        self.n_splits = n_splits
        self.seed = seed

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        x_indices = np.arange(n_samples)

        # get distribution of labels
        labels, indices, counts = np.unique(y._.values if isinstance(y, (pd.Series, pd.DataFrame)) else y, axis=0, return_inverse=True, return_counts=True)
        required_samples = counts.max()

        with temp_seed(self.seed):
            # randomize each bucket and split each randomized bucket into n_splits
            sample_indices_per_label = {l: np.random.permutation(x_indices[indices == l]) for l in range(len(labels))}
            for l, b in sample_indices_per_label.items():
                # if we do not have enough samples we need to allow replace
                replace = (required_samples - len(b)) >= len(b)
                sample_indices_per_label[l] = np.hstack([b, np.random.choice(b, required_samples - len(b), replace=replace)])

            # resample all events
            resampled_indices = np.hstack(list(sample_indices_per_label.values()))
            resampled_labels = np.hstack([[l] * len(b) for l, b in sample_indices_per_label.items()])

            # to avoid sorted labels we need to shuffle again
            shuffled_indices = np.random.permutation(np.arange(len(resampled_indices)))
            resampled_indices = resampled_indices[shuffled_indices]
            resampled_labels = resampled_labels[shuffled_indices]

        # return StratifiedKFold
        for train_idx, test_idx in StratifiedKFold(self.n_splits, shuffle=False).split(resampled_indices, resampled_labels):
            yield resampled_indices[train_idx], resampled_indices[test_idx]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class PartitionedOnRowMultiIndexCV(object):

    def __init__(self, cross_validation):
        self.delegated = cross_validation.split if hasattr(cross_validation, "split") else cross_validation

    def split(self, x, *args, **kwargs) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if isinstance(x, pd.MultiIndex):
            grp_offset = 0
            for group in unique_level_rows(x):
                grp_args = [v.loc[group] if isinstance(v, PandasObject) else v for v in args]
                grp_kwargs = {k: v.loc[group] if isinstance(v, PandasObject) else v for k, v in kwargs.items()}
                for train, test in call_callable_dynamic_args(self.delegated, x[x.get_loc(group)], *grp_args, **grp_kwargs):
                    yield train + grp_offset, test + grp_offset

                grp_offset += len(x[x.get_loc(group)])
        else:
            return call_callable_dynamic_args(self.delegated, x, **kwargs)