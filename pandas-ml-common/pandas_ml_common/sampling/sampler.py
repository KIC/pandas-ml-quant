import logging
from typing import Callable, Tuple, Generator, List, Union

import numpy as np
import pandas as pd

from pandas_ml_common.utils import unpack_nested_arrays
from pandas_ml_common.utils.index_utils import intersection_of_index, loc_if_not_none, unique_level_rows

_log = logging.getLogger(__name__)


class Sampler(object):

    def __init__(
            self,
            *dataframes: pd.DataFrame,
            splitter: Callable[[pd.Index, pd.DataFrame], Tuple[pd.Index, pd.Index]],
            training_samples_filter: Union['BaseCrossValidator', Tuple[int, Callable[[pd.Series], bool]]] = None,
            cross_validation: Tuple[int, Callable[[pd.Index], Tuple[List[int], List[int]]]] = None,
            epochs: int = 1
    ):
        """
        This Class is used to sample from DataFrames training and test data sets

        :param dataframes: a lit of dataframes like features, labels, sample weights, ...
        :param splitter: a function which gets the intersection index of all passed data frames and returns the train
            and test indices
        :param training_samples_filter: a function which might filter out data from the training set on a given dataframe
            index. This is useful i.e. to filter out unclear cases like data on the border of a bucket
        :param cross_validation: a tuple with the number of folds and a function which splits the training data index
            further into number of folds
        :param epochs:
            the number of epochs tells the generator how often the dataset split will be yielded before the generator
            ends. None can ne passed as well which leads to an infinite generator useful for reinforceent learning
        """
        self.common_index = intersection_of_index(*dataframes)
        self.frames = [loc_if_not_none(f, self.common_index) for f in dataframes]
        self.training_samples_filter_frame = training_samples_filter[0] if training_samples_filter is not None else None
        self.training_samples_filter = training_samples_filter[1] if training_samples_filter is not None else None
        if cross_validation is not None:
            if hasattr(cross_validation, 'get_n_splits') and hasattr(cross_validation, 'split'):
                self.cross_validation_nr_folds = cross_validation.get_n_splits()
                self.cross_validation = cross_validation.split
            elif isinstance(cross_validation, Tuple):
                self.cross_validation_nr_folds = cross_validation[0]
                self.cross_validation = cross_validation[1]
            else:
                raise ValueError("Expected scikit BaseCrossValidator or Tuple[nr_of_folds, splitter()]")
        else:
            self.cross_validation_nr_folds = 0
            self.cross_validation = None

        self.splitter = splitter
        self.epochs = epochs

        if self.cross_validation is not None and not isinstance(self.epochs, int):
            raise ValueError(f"epochs need to be finite integer in case of cross validation: {self.epochs}")

    @property
    def nr_of_cross_validation_folds(self):
        return self.cross_validation_nr_folds

    def sample_cross_validation(
            self,
            on_epoch: Callable = None,
            on_fold: Callable = None
    ) -> Generator[Tuple[int, int, List[pd.DataFrame], List[pd.DataFrame]], None, None]:
        """
        Samples training data and test data from the given data frames using cross validation. For each fold
        a separate model should be trained. All fold models then might be averaged together or only the best one might
        be used.

        :param on_epoch: optional callback called before each epoch starts
        :param on_fold: optional callback called before each fold starts
        :return: returns a generator of a tuple(nr of fold, training data, val data, test data)
        """

        if self.cross_validation is None:
            # simply wrapping the sample generator to return the same data structure as if it would be cross validated
            for epoch, train_data, test_data in self.sample(on_epoch):
                yield epoch, -1, train_data, test_data
        else:
            # here we only need one epoch for each frame!
            row_frame_folds = [(train_idx, test_idx, list(enumerate(self.cross_validation(train_idx, unpack_nested_arrays(self.frames[1].loc[train_idx].values)))))
                               for epoch, train_idx, test_idx in self._sample(self.common_index, 1, False, None)]

            # loop epochs
            for epoch in range(self.epochs):
                if callable(on_epoch):
                    on_epoch()

                # in each epoch loop all frames
                for train_idx, test_idx, folds in row_frame_folds:
                    # and within each frame loop all folds. Just looping one fold after the other is not very useful
                    # but reflects the current behaviour. It would be better to train different models for each fold.
                    # Therefore wre to loop folds is less important.
                    for f, (cv_train_idx, cv_val_idx) in folds:
                        if callable(on_fold):
                            on_fold()

                        yield (
                            epoch,
                            f,
                            [loc_if_not_none(f, train_idx[cv_train_idx]) for f in self.frames],
                            [loc_if_not_none(f, train_idx[cv_val_idx]) for f in self.frames]
                        )

                    yield (
                        epoch,
                        -1,
                        [loc_if_not_none(f, train_idx) for f in self.frames],
                        [loc_if_not_none(f, test_idx) for f in self.frames]
                    )

    def sample_full_epochs(self) -> Generator[Tuple[int, List[pd.DataFrame], List[pd.DataFrame]], None, None]:
        return self.sample(full_epoch=True)

    def sample(self, full_epoch: bool = False, on_epoch: Callable = None) -> Generator[Tuple[int, List[pd.DataFrame], List[pd.DataFrame]], None, None]:
        """
        Samples training data and test data from the given data frames

        :param full_epoch: enforce a full epoch which keeps the multi index row otherwiese we get a sample for each
            epoch and for each top level row in a multi index row data frame
        :param on_epoch: optional callback called before each epoch starts
        :return: returns a generator of a tuple( training data, test data)
        """
        for epoch, train_idx, test_idx in self._sample(self.common_index, self.epochs, full_epoch, on_epoch):
            yield (
                epoch,
                [loc_if_not_none(f, train_idx) for f in self.frames],
                [loc_if_not_none(f, test_idx) for f in self.frames]
            )

    def _sample(self, idx: pd.Index, epochs, full_epochs, on_epoch) -> Generator[Tuple[int, pd.Index, pd.Index], None, None]:
        # lazy initialize the training and test index arrays
        train_idx, test_idx = (None, None)

        # loop for all epochs there is a way to infinitely loop epoch which is needed i.e. for reinforcement learning
        for epoch in (range(epochs) if epochs is not None else iter(int, 1)):
            if callable(on_epoch):
                on_epoch()

            if isinstance(idx, pd.MultiIndex):
                train_test = (pd.Index([]), pd.Index([])) if full_epochs else None

                for group in unique_level_rows(idx):
                    grp_idx = idx[idx.get_loc(group)].to_flat_index()
                    for _, train, test in self._sample(grp_idx, 1, False, None):
                        if not full_epochs:
                            yield epoch, train, test
                        else:
                            train_test = (train_test[0].append(train), train_test[1].append(test))

                if train_test is not None:
                    yield (epoch, *train_test)

            else:
                if train_idx is None:
                    # lazily initialize the training test data split
                    if self.splitter is None:
                        if self.training_samples_filter is not None:
                            _log.warning("training_samples_filter is not None but no test index is defined ... skipped")

                        train_idx, test_idx = (idx, [])
                    else:
                        train_idx, test_idx = self.splitter(idx, self.frames[1])

                        if self.training_samples_filter is not None:
                            df = self.frames[self.training_samples_filter_frame].loc[train_idx]
                            train_idx = train_idx[df.apply(self.training_samples_filter, axis=1).values]

                yield epoch, train_idx, test_idx


class NumpySampler(object):

    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    @property
    def nr_of_cross_validation_folds(self):
        return self.sampler.nr_of_cross_validation_folds

    def sample_cross_validation(
            self,
            on_epoch: Callable = None,
            on_fold: Callable = None
    ) -> Generator[Tuple[int, int, pd.Index, List[np.ndarray], pd.Index, List[np.ndarray]], None, None]:
        for epoch, fold, train, test in self.sampler.sample_cross_validation(on_epoch=on_epoch, on_fold=on_fold):
            yield (
                epoch,
                fold,
                train[0].index,
                tuple([NumpySampler._to_numpy(t) for t in train]),
                test[0].index,
                tuple([NumpySampler._to_numpy(t) for t in test])
            )

    def sample_full_epochs(self) -> Generator[Tuple[int, List[np.ndarray], List[np.ndarray]], None, None]:
        for epoch, train, test in self.sampler.sample_full_epochs():
            yield (
                epoch,
                tuple([NumpySampler._to_numpy(t) for t in train]),
                tuple([NumpySampler._to_numpy(t) for t in test])
            )

    def sample(self, full_epoch: bool = False, on_epoch: Callable = None) -> Generator[Tuple[int, List[np.ndarray], List[np.ndarray]], None, None]:
        for epoch, train, test in self.sampler.sample(full_epoch=full_epoch, on_epoch=on_epoch):
            yield (
                epoch,
                tuple([NumpySampler._to_numpy(t) for t in train]),
                tuple([NumpySampler._to_numpy(t) for t in test])
            )

    @staticmethod
    def _to_numpy(df):
        if df is None or len(df) <= 0:
            return None

        values = df._.values

        if isinstance(values, list):
            return np.concatenate(values, axis=0)
        else:
            return values