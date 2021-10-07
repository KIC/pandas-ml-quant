from abc import abstractmethod
import logging
from abc import abstractmethod
from functools import partial
from typing import Callable, Optional, Tuple, List

import numpy as np
import pandas as pd

from pandas_ml_common import MlTypes, FeaturesLabels, call_callable_dynamic_args, LazyInit
from pandas_ml_common.preprocessing.features_labels import FeaturesWithReconstructionTargets
from pandas_ml_common.sampling.sampler import FoldXYWeight, XYWeight
from pandas_ml_common.trainingloop import sampling
from pandas_ml_common.utils import merge_kwargs, to_pandas
from pandas_ml_utils.ml.fitting.fitting_parameter import FittingParameter
from .predictable import Model, SubModelFeature
from ..data.reconstruction import assemble_result_frame
from ..fitting import FitStatistics
from ..forecast import Forecast
from ..summary import Summary

_log = logging.getLogger(__name__)


class Fittable(Model):

    def __init__(self,
                 features_and_labels_definition: FeaturesLabels,
                 summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = Summary,
                 forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                 **kwargs):
        super().__init__(features_and_labels_definition, forecast_provider, **kwargs)
        self._summary_provider = summary_provider
        self._fit_statistics: Optional[FitStatistics] = None

    @property
    def summary_provider(self):
        return self._summary_provider

    @property
    def fit_statistics(self) -> FitStatistics:
        return self._fit_statistics

    def fit_to_df(self,
            df: MlTypes.PatchedDataFrame,
            fitting_parameter: FittingParameter,
            verbose: int = 0,
            callbacks: Optional[Callable[..., None]] = None,
            **kwargs) -> Tuple[MlTypes.PatchedDataFrame, MlTypes.PatchedDataFrame]:
        typemap_fitting = {SubModelFeature: lambda df, model, **kwargs: model.fit(df, **kwargs)}
        merged_kwargs = merge_kwargs(self.kwargs, kwargs)

        # initialize the fit of the model
        self.init_fit(fitting_parameter, **merged_kwargs)
        processed_batches = 0

        # set up a sampler for the data
        frames, sampler = sampling(
            df=df,
            features_and_labels_definition=self.features_and_labels_definition,
            type_mapping=typemap_fitting,
            splitter=fitting_parameter.splitter,
            filter=fitting_parameter.filter,
            cross_validation=fitting_parameter.cross_validation,
            epochs=fitting_parameter.epochs,
            fold_epochs=fitting_parameter.fold_epochs,
            batch_size=fitting_parameter.batch_size,
            partition_row_multi_index_batch=fitting_parameter.partition_row_multi_index_batch,
            **merged_kwargs
        )

        # extract the used training and test data
        training_data = frames.features_with_required_samples.loc[sampler.get_in_sample_features_index]
        test_data = frames.features_with_required_samples.loc[sampler.get_out_of_sample_features_index]

        # remember min required samples and label names
        self.min_required_samples = frames.features_with_required_samples.min_required_samples
        # TODO we can have multiple labes so we have a list of list like List[List[column_names]]
        self._label_names = [(i, col) if len(frames.labels) > 1 else col for i, l in enumerate(frames.labels) for col in l]
        self._fit_statistics = FitStatistics(fitting_parameter)

        # register the fitting logic callbacks
        sampler = sampler.with_callbacks(
            on_start=partial(self._init_fit, fitting_parameter=fitting_parameter, **merged_kwargs),
            on_fold=partial(self.init_fold, fitting_parameter=fitting_parameter, **merged_kwargs),
            after_fold_epoch=partial(self._after_fold_epoch, callbacks=callbacks, verbose=verbose, training_data=training_data, testing_data=test_data, **merged_kwargs),
            after_epoch=partial(self._after_epoch, callbacks=callbacks, verbose=verbose, **merged_kwargs),
            after_end=partial(self.finish_learning, **merged_kwargs)
        )

        # fit the model
        for batch in sampler.sample_for_training():
            self.fit_batch(batch, **merged_kwargs)
            processed_batches += 1

        # check if we have fitted the model at all
        if processed_batches <= 0:
            raise ValueError(f"Not enough data {[len(f) for f in sampler.frames[0]]}")

        # reconstruct the used training and test DataFrmes
        df_training_prediction = \
            to_pandas(self.train_predict(training_data, **merged_kwargs), training_data.common_index, self.label_names)

        if len(test_data.common_index) > 0:
            df_test_prediction = \
                to_pandas(self.predict(test_data, **merged_kwargs), test_data.common_index, self.label_names)
        else:
            df_test_prediction = pd.DataFrame({}, columns=self.label_names)

        # assemble the result frames and return the result
        feature_frames = frames.features_with_required_samples
        label_frames = frames.labels_with_sample_weights
        ext_frames = (
            feature_frames.reconstruction_targets,
            label_frames.joint_label_frame, label_frames.gross_loss, label_frames.joint_sample_weights_frame,
            feature_frames.joint_feature_frame
        )

        return (
            assemble_result_frame(df_training_prediction, *ext_frames),
            assemble_result_frame(df_test_prediction[~df_test_prediction.index.duplicated()], *ext_frames)
        )

    def _init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        self.init_fit(fitting_parameter, **kwargs)

    def _after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight], training_data: FeaturesWithReconstructionTargets, testing_data: FeaturesWithReconstructionTargets, verbose, callbacks, **kwargs):
        # calculate the training and test losses
        train_loss, test_loss = self.calculate_train_test_loss(fold, train_data, test_data, **kwargs)
        if verbose:
            print(f"loss training data: {train_loss}, test data: {test_loss}")

        # record statistics
        self._fit_statistics.record_loss(epoch, fold, fold_epoch, train_loss, test_loss)

        # execute callbacks
        call_callable_dynamic_args(
            callbacks,
            epoch=epoch, fold=fold, fold_epoch=fold_epoch, loss=train_loss, test_loss=test_loss, val_loss=test_loss,
            y_train=train_data.y, y_test=[td.y for td in test_data],
            y_hat_train=LazyInit(lambda: self.predict(training_data)),
            y_hat_test=LazyInit(lambda: self.predict(testing_data))
        )

        self.after_fold_epoch(epoch, fold, fold_epoch, train_data, test_data, **kwargs)

    def _after_epoch(self, epoch: int, verbose, callbacks, **kwargs):
        self.after_epoch(epoch, **kwargs)

    @abstractmethod
    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        pass

    @abstractmethod
    def init_fold(self, epoch: int, fold: int, fitting_parameter: FittingParameter, **kwargs):
        pass

    @abstractmethod
    def fit_batch(self, xyw: FoldXYWeight, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight], **kwargs):
        pass

    @abstractmethod
    def after_epoch(self, epoch: int, **kwargs):
        pass

    @abstractmethod
    def train_predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def calculate_train_test_loss(self, fold: int, train_data: XYWeight, test_data: List[XYWeight], **kwargs) -> MlTypes.Loss:
        pass

    @abstractmethod
    def finish_learning(self, **kwargs):
        pass

