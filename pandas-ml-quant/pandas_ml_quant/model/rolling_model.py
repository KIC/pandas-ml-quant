from functools import partial
from typing import List

import pandas as pd
from sortedcontainers import SortedDict

from pandas_ml_common import Typing, Sampler, XYWeight
from pandas_ml_common.sampling.cross_validation import PartitionedOnRowMultiIndexCV
from pandas_ml_common.sampling.splitter import duplicate_data
from pandas_ml_utils import Model
from .cross_validation.rolling_window_cv import RollingWindowCV


class RollingModel(Model):

    def __init__(self, delegated_model: Model, window: int = 200, retrain_after: int = 7, **kwargs):
        super().__init__(delegated_model.features_and_labels, delegated_model.summary_provider, **{**delegated_model.kwargs, **kwargs})
        self.delegated_model = delegated_model
        self.cross_validation = RollingWindowCV(window, retrain_after)
        self._used_folds = SortedDict()  # using bisect_left gives us the ceiling iloc
        self._past_predictions = []
        self._past_train_predictions = []

    def _sampler_with_callbacks(self, sampler: Sampler, verbose: int = 0, callbacks=None) -> Sampler:
        return sampler.with_callbacks(
            on_start=self._record_meta,
            on_fold=self.init_fold,
            after_fold_epoch=partial(self._record_loss, callbacks=callbacks, verbose=verbose),
            after_fold=self.merge_folds,
            after_end=self.finish_learning
        )

    def to_fitter_kwargs(self, partition_row_multi_index=False, **kwargs):
        return {
            "model_provider": self,
            "splitter": duplicate_data(),
            "cross_validation": PartitionedOnRowMultiIndexCV(self.cross_validation) if partition_row_multi_index else self.cross_validation,
            **kwargs
        }

    @property
    def features_and_labels(self):
        return self.delegated_model.features_and_labels

    @property
    def summary_provider(self):
        return self.delegated_model.summary_provider

    def _record_meta(self, epochs, batch_size, fold_epochs, cross_validation, features, labels: List[str]):
        self.delegated_model._record_meta(epochs, batch_size, fold_epochs, cross_validation, features, labels)

    def _record_loss(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight], verbose, callbacks, loss_history_key=0):
        self.delegated_model._record_loss(epoch, fold, fold, train_data, test_data, verbose, callbacks, "rolling")

    def init_fit(self, **kwargs):
        self.delegated_model.init_fit(**kwargs)

    def init_fold(self, epoch: int, fold: int):
        self.delegated_model.init_fold(epoch, fold)

    def fit_batch(self, x: pd.DataFrame, y: pd.DataFrame, w: pd.DataFrame, fold: int, **kwargs):
        self.delegated_model.fit_batch(x, y, w, fold, **kwargs)

    def after_fold_epoch(self, epoch, fold, fold_epoch, loss, val_loss):
        self.delegated_model.after_fold_epoch(epoch, fold, fold_epoch, loss, val_loss)

    def calculate_loss(self, fold: int, x: pd.DataFrame, y_true: pd.DataFrame, weight: pd.DataFrame) -> float:
        return self.delegated_model.calculate_loss(fold, x, y_true, weight)

    def merge_folds(self, epoch: int, fold, train_data, test_data):
        self.delegated_model.merge_folds(epoch)

        # remember test predictions
        self._past_train_predictions.append(self.delegated_model.predict(train_data.x[-self.cross_validation.retrain_after:]))

        # remember past predictions as kind of cache, otherwise we would need to keep thousands of models in memory
        # if we would want to know the feature importance we would need to calculate them here
        max_indexes = []
        for i, td in enumerate(test_data):
            if len(td.x) <= 0:
                continue

            y_hat = self.delegated_model.predict(td.x)
            if len(test_data) > 1:
                y_hat.columns = pd.MultiIndex.from_product([[i], y_hat.columns])

            self._past_predictions.append(y_hat)
            max_indexes.append(td.x.index[-1])

        if len(max_indexes) > 0:
            self._used_folds[max(max_indexes)] = fold

    def train_predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        return pd.concat(self._past_train_predictions, axis=0)

    def predict(self, features: pd.DataFrame, targets: pd.DataFrame = None, latent: pd.DataFrame = None, samples=1, **kwargs) -> Typing.PatchedDataFrame:
        if len(self._past_predictions) <= 0:
            raise ValueError("No validation data present, Model need to be trained first or more data is needed")

        past_predictions = pd.concat(self._past_predictions, axis=0)
        last_prediction_index = past_predictions.index[-1]
        index = features.index

        if last_prediction_index not in index:
            raise ValueError(f"Data gapped away, passed features data need to start at {last_prediction_index}")

        unpredicted_features_index = index[index.get_loc(last_prediction_index):]
        if len(unpredicted_features_index) > max(self.cross_validation.retrain_after, 1):
            # TODO eventually we can just retrain the model automatically
            #  from unpredicted_features_index[-self.cross_validation.retrain_after:] onwards
            raise ValueError(f"Model need to be re-trained! {past_predictions.index[-1]}, {index[-1]}")

        def predictor(loc):
            if loc in past_predictions.index:
                return past_predictions.loc[[loc]]
            else:
                return self.delegated_model.predict(features.loc[[loc]], targets, latent, samples, **kwargs)

        return pd.concat([predictor(idx) for idx in features.index if idx >= past_predictions.index[0]], axis=0)

    def finish_learning(self):
        self.delegated_model.finish_learning()

    def plot_loss(self, figsize=(8, 6), **kwargs):
        return self.delegated_model.plot_loss(figsize, **kwargs)
