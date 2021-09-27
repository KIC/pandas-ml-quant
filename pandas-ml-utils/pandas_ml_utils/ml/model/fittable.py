from abc import ABCMeta
from abc import abstractmethod
from functools import partial
from typing import Dict, Type, Callable, Optional, Tuple, List, Any
from typing import Union
import logging
import numpy as np
import pandas as pd

from pandas_ml_common import MlTypes, FeaturesLabels
from pandas_ml_common.preprocessing.features_labels import FeaturesWithLabels, FeaturesWithReconstructionTargets
from pandas_ml_common.sampling.sampler import FoldXYWeight, XYWeight
from pandas_ml_common.trainingloop import sampling
from pandas_ml_common.utils import merge_kwargs, none_as_empty_dict, to_pandas
from pandas_ml_utils.constants import PREDICTION_COLUMN_NAME
from pandas_ml_utils.ml.fitting.fitting_parameter import FittingParameter
from .persistence import Model, SubModelFeature
from ..data.reconstruction import assemble_result_frame
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

    @property
    def summary_provider(self):
        return self._summary_provider

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
            batch_size=fitting_parameter.batch_size
        )

        # remember min required samples and label names
        self.min_required_samples = frames.features_with_required_samples.min_required_samples
        self._label_names = [(i, col) for i, l in enumerate(frames.labels) for col in l]

        # register the fitting logic callbacks
        sampler = sampler.with_callbacks(
            on_start=partial(self._init_fit, fitting_parameter=fitting_parameter),
            on_fold=self.init_fold,
            after_fold_epoch=partial(self._after_fold_epoch, callbacks=callbacks, verbose=verbose),
            after_epoch=partial(self._after_epoch, callbacks=callbacks, verbose=verbose),
            after_end=self.finish_learning
        )

        # fit the model
        for batch in sampler.sample_for_training():
            self.fit_batch(batch, **merged_kwargs)
            processed_batches += 1

        # chceck if we have fitted the model at all
        if processed_batches <= 0:
            raise ValueError(f"Not enough data {[len(f) for f in sampler.frames[0]]}")

        # extract the used training data
        training_data = frames.features_with_required_samples.loc[sampler.get_in_sample_features_index]
        df_training_prediction = \
            to_pandas(self.train_predict(training_data, **merged_kwargs), training_data.common_index, self.label_names)

        # extract the used test data
        test_data = frames.features_with_required_samples.loc[sampler.get_out_of_sample_features_index]
        if len(test_data) > 0:
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

    def _after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight], verbose, callbacks):
        # # FIXME ... remimplemt all this and introduce a FitStaistics object
        # train_loss, test_loss = self.calculate_train_test_loss(epoch, fold, train_data, test_data, verbose)
        # self._statistics.record_loss(loss_history_key or fold, epoch, fold, fold_epoch, train_loss, test_loss)
        #
        # call_callable_dynamic_args(
        #     callbacks,
        #     epoch=epoch, fold=fold, fold_epoch=fold_epoch, loss=train_loss, test_loss=test_loss, val_loss=test_loss,
        #     y_train=train_data.y, y_test=[td.y for td in test_data],
        #     y_hat_train=LazyInit(lambda: self.predict(train_data.x)),
        #     y_hat_test=[LazyInit(lambda: self.predict(td.x)) for td in test_data]
        # )
        pass

    def _after_epoch(self, epoch: int, verbose, callbacks):
        self.after_epoch(epoch)

    @abstractmethod
    def init_fit(self, fitting_parameter: FittingParameter, **kwargs):
        pass

    @abstractmethod
    def init_fold(self, epoch: int, fold: int):
        pass

    @abstractmethod
    def fit_batch(self, xyw: FoldXYWeight, **kwargs) -> MlTypes.Loss:
        raise NotImplementedError()

    @abstractmethod
    def after_fold_epoch(self, epoch, fold, fold_epoch, train_data: XYWeight, test_data: List[XYWeight]):
        pass

    @abstractmethod
    def after_epoch(self, epoch: int):
        pass

    @abstractmethod
    def train_predict(self, features: FeaturesWithReconstructionTargets, samples: int = 1, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def finish_learning(self, **kwargs):
        pass


