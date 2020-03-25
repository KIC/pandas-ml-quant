import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import _num_samples


class KFoldBoostRareEvents(KFold):

    def __init__(self, n_splits='warn', shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle, random_state)

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        # all events which are true are considered to be rare
        rare_event_indices = indices[y.ravel() >= 0.999]

        for f, (train_idx, test_idx) in enumerate(super().split(X, y, groups)):
            yield np.hstack([train_idx, rare_event_indices]), np.hstack([test_idx, rare_event_indices])


class KEquallyWeightEvents(object):

    def __init__(self, n_splits='warn', seed=None):
        super().__init__()
        self.n_splits = n_splits
        np.random.seed(seed)

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        x_indices = np.arange(n_samples)

        # get distribution of labels
        labels, indices, counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
        required_samples = counts.max()

        # randomize each bucket and split each randomized bucket into n_splits
        sample_indices_per_label = {l: np.random.permutation(x_indices[indices == l]) for l in range(len(labels))}
        for l, b in sample_indices_per_label.items():
            # if we do not have enough samples we need to allow replace
            replace = (required_samples - len(b)) >= len(b)
            sample_indices_per_label[l] = np.hstack([b, np.random.choice(b, required_samples - len(b), replace=replace)])

        # resample all events
        resampled_indices = np.hstack(sample_indices_per_label.values())
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
