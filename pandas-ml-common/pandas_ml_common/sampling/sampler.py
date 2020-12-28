import logging
from typing import Tuple, Callable, Any, Generator, NamedTuple, Union, List

import numpy as np
import pandas as pd

from pandas_ml_common.sampling.cross_validation import PartitionedOnRowMultiIndexCV
from pandas_ml_common.utils import call_callable_dynamic_args, intersection_of_index, loc_if_not_none, exec_if_not_none

_log = logging.getLogger(__name__)


class XYWeight(NamedTuple):
    x: pd.DataFrame
    y: pd.DataFrame = None
    weight: pd.DataFrame = None

    def to_dict(self, loc=None):
        d = {"x": self.x, "y": self.y, "weight": self.weight}
        if loc is not None:
            d = {k: loc_if_not_none(v, loc) for k, v in d.items()}

        return d


class FoldXYWeight(NamedTuple):
    epoch: int
    fold: int
    epoch_fold: int
    x: pd.DataFrame
    y: pd.DataFrame = None
    weight: pd.DataFrame = None


class Sampler(object):

    def __init__(
            self,
            frames: XYWeight,
            splitter: Callable[[Any], Tuple[pd.Index, pd.Index]] = None,
            filter: Callable[[Any], bool] = None,
            cross_validation: Union['BaseCrossValidator', Callable[[Any], Generator[Tuple[np.ndarray, np.ndarray], None, None]]] = None,
            epochs: int = 1,
            batch_size: int = None,
            fold_epochs: int = 1,
            on_start: Callable = None,
            on_epoch: Callable = None,
            on_batch: Callable = None,
            on_fold: Callable = None,
            on_fold_epoch: Callable = None,
            after_epoch: Callable = None,
            after_batch: Callable = None,
            after_fold: Callable = None,
            after_fold_epoch: Callable = None,
            after_end: Callable = None,
            **kwargs
    ):
        self.common_index = intersection_of_index(*frames).sort_values()
        self.frames = XYWeight(*[loc_if_not_none(f, self.common_index) for f in frames])
        self.epochs = epochs
        self.batch_size = batch_size
        self.fold_epochs = fold_epochs
        self.splitter = splitter
        self.filter = filter

        # callbacks
        self.on_start = on_start
        self.on_epoch = on_epoch
        self.on_batch = on_batch
        self.on_fold = on_fold
        self.on_fold_epoch = on_fold_epoch
        self.after_epoch = after_epoch
        self.after_batch = after_batch
        self.after_fold = after_fold
        self.after_fold_epoch = after_fold_epoch
        self.after_end = after_end

        # split training and test data
        if self.splitter is not None:
            if isinstance(self.common_index, pd.MultiIndex):
                _log.warning("The Data provided uses a `MultiIndex`, eventually you want to set the "
                             "`partition_row_multi_index` parameter in your splitter")

            self.train_idx, self.test_idx = call_callable_dynamic_args(
                self.splitter, self.common_index, **self.frames.to_dict())
        else:
            self.train_idx, self.test_idx = self.common_index, pd.Index([])

        if cross_validation is not None:
            if isinstance(self.common_index, pd.MultiIndex) and not isinstance(cross_validation, PartitionedOnRowMultiIndexCV):
                # cross validators need to fold within each group of a multi index row index, a wrapper can be provided
                _log.warning("The Data provided uses a `MultiIndex` but the cross validation is not wrapped in "
                             "`PartitionedOnRowMultiIndexCV`")

            if epochs is None or epochs > 1:
                _log.warning(f"using epochs > 1 together with cross folding may lead to different folds for each epoch!"
                             f"{cross_validation}")

            self.nr_folds = cross_validation.get_n_splits() if hasattr(cross_validation, "get_n_splits") else -1
            self.cross_validation = cross_validation.split if hasattr(cross_validation, "split") else cross_validation
        else:
            self.nr_folds = None
            self.cross_validation = None

    def with_callbacks(
            self,
            on_start: Callable = None,
            on_epoch: Callable = None,
            on_batch: Callable = None,
            on_fold: Callable = None,
            on_fold_epoch: Callable = None,
            after_epoch: Callable = None,
            after_batch: Callable = None,
            after_fold: Callable = None,
            after_fold_epoch: Callable = None,
            after_end: Callable = None,
    ):
        return Sampler(
            self.frames,
            self.splitter,
            self.filter,
            self.cross_validation,
            self.epochs,
            self.batch_size,
            self.fold_epochs,
            on_start,
            on_epoch,
            on_batch,
            on_fold,
            on_fold_epoch,
            after_epoch,
            after_batch,
            after_fold,
            after_fold_epoch,
            after_end
        )

    def sample_for_training(self) -> Generator[FoldXYWeight, None, None]:
        cross_validation = self.cross_validation if self.cross_validation is not None else lambda x: [(None, None)]

        # filter samples
        if self.filter is not None:
            train_idx = [idx for idx in self.train_idx if call_callable_dynamic_args(self.filter, idx, **self.frames.to_dict(idx))]
        else:
            train_idx = self.train_idx

        # update frame views
        train_frames = XYWeight(*[loc_if_not_none(f, train_idx) for f in self.frames])
        test_frames = XYWeight(*[loc_if_not_none(f, self.test_idx) for f in self.frames])

        # call for start ...
        call_callable_dynamic_args(
            self.on_start,
            epochs=self.epochs, batch_size=self.batch_size, fold_epochs=self.fold_epochs,
            features=exec_if_not_none(lambda x: x.columns.tolist(), self.frames.x),
            labels=exec_if_not_none(lambda y: y.columns.tolist(), self.frames.y),
            cross_validation=self.nr_folds is not None)

        # generate samples
        for epoch in (range(self.epochs) if self.epochs is not None else iter(int, 1)):
            call_callable_dynamic_args(self.on_epoch, epoch=epoch)
            fold_iter = enumerate(call_callable_dynamic_args(cross_validation, train_idx, **train_frames.to_dict()))
            for fold, (cv_train_i, cv_test_i) in fold_iter:
                call_callable_dynamic_args(self.on_fold, epoch=epoch, fold=fold)

                # if we dont have any cross validation the training and test sets stay unchanged
                cv_train_idx = train_idx if cv_train_i is None else train_idx[cv_train_i]

                # build our test data sets
                if cv_test_i is not None:
                    if cv_test_i.ndim > 1:
                        cv_test_frames = [
                            XYWeight(*[loc_if_not_none(f, train_idx[cv_test_i[:, i]]) for f in self.frames])
                            for i in range(cv_test_i.shape[1])
                        ]
                    else:
                        cv_test_frames = [
                            XYWeight(*[loc_if_not_none(f, train_idx[cv_test_i]) for f in self.frames])]
                else:
                    if len(self.test_idx) <= 0:
                        cv_test_frames = []
                    else:
                        cv_test_frames = [XYWeight(*[loc_if_not_none(f, self.test_idx) for f in self.frames])]

                for fold_epoch in range(self.fold_epochs):
                    call_callable_dynamic_args(self.on_fold, epoch=epoch, fold=fold, fold_epoch=fold_epoch)

                    # build our training data sets aka batches
                    cv_train_frames = XYWeight(*[loc_if_not_none(f, cv_train_idx) for f in self.frames])

                    # theoretically we could already yield cv_train_frames, cv_test_frames
                    # but lets create batches first and then yield all together
                    nr_instances = len(cv_train_idx)
                    nice_i = max(nr_instances - 2, 0)
                    bs = min(nr_instances, self.batch_size) if self.batch_size is not None else nr_instances

                    batch_iter = range(0, nr_instances, bs)
                    for i in batch_iter:
                        call_callable_dynamic_args(self.on_batch, epoch=epoch, fold=fold, fold_epoch=fold_epoch, batch=i)
                        yield FoldXYWeight(epoch, fold, fold_epoch, *(f.iloc[i if i < nice_i else i - 1:i + bs] if f is not None else None for f in cv_train_frames))
                        call_callable_dynamic_args(self.after_batch, epoch=epoch, fold=fold, fold_epoch=fold_epoch, batch=i)

                    # end of fold epoch
                    try:
                        call_callable_dynamic_args(self.after_fold_epoch, epoch=epoch, fold=fold, fold_epoch=fold_epoch, train_data=cv_train_frames, test_data=cv_test_frames)
                    except StopIteration as sie:
                        call_callable_dynamic_args(self.after_fold, epoch=epoch, fold=fold, train_data=cv_train_frames, test_data=cv_test_frames)

                        if str(sie).isnumeric() and int(str(sie)) == fold:
                            # we just want to stop this fold
                            break
                        else:
                            # we need to stop any further generation of sample and call all left callbacks
                            call_callable_dynamic_args(self.after_epoch, epoch=epoch, train_data=train_frames, test_data=test_frames)
                            call_callable_dynamic_args(self.after_end)
                            return
                # end of fold
                call_callable_dynamic_args(self.after_fold, epoch=epoch, fold=fold, train_data=cv_train_frames, test_data=cv_test_frames)
            # end of epoch
            call_callable_dynamic_args(self.after_epoch, epoch=epoch, train_data=train_frames, test_data=test_frames)
        # end of generator
        call_callable_dynamic_args(self.after_end)

    def get_in_sample_features(self) -> pd.DataFrame:
        return self.frames.x.loc[self.train_idx]

    def get_out_of_sample_features(self) -> pd.DataFrame:
        return self.frames.x.loc[self.test_idx]

