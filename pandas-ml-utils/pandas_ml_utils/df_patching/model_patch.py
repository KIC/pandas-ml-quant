import logging
from time import perf_counter
from typing import Callable, Union, List, Type

from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold

from pandas_ml_common import Typing, naive_splitter, XYWeight
from pandas_ml_common.utils import get_correlation_pairs, merge_kwargs, call_silent, \
    call_callable_dynamic_args
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels, \
    PostProcessedFeaturesAndLabels
from pandas_ml_utils.ml.data.reconstruction import assemble_result_frame
from pandas_ml_utils.ml.fitting import Fit, FitException, FittingParameter
from pandas_ml_utils.ml.forecast import Forecast
from pandas_ml_utils.ml.model import Model as MlModel, SkModel, AutoEncoderModel, SubModelFeature
from pandas_ml_utils.ml.summary import Summary
from ..ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels, FeaturesWithTargets, \
    extract_features, extract_feature_labels_weights
from ..ml.summary.feature_selection_summary import FeatureSelectionSummary

_log = logging.getLogger(__name__)


class DfModelPatch(object):

    def __init__(self, df: Typing.PatchedDataFrame):
        self.df = df
        self._type_mapping = {
            AutoEncoderModel: lambda df, model, **kwargs: df.model.predict(model.as_encoder(), **kwargs),
            MlModel: lambda df, model, **kwargs: df.model.predict(model, **kwargs),
        }

    def feature_selection(self,
                          features_and_labels: FeaturesAndLabels,
                          training_data_splitter: Callable = naive_splitter(0.2),
                          correlated_features_th: float = 0.75,
                          rfecv_splits: int = 4,
                          forest_splits: int = 7,
                          min_features_to_select: int = 1,
                          is_time_series: bool = False,
                          **kwargs):
        assert features_and_labels.label_type in ('regression', 'classification', int, float, bool), \
            "label_type need to be specified: 'regression' | 'classification' !"

        # find best parameters
        with self() as m:
            # extract features and labels
            ext = m.extract(features_and_labels)

            # first perform a correlation analysis and remove correlating features !!!
            if correlated_features_th > 0:
                _, pairs = get_correlation_pairs(ext.features)
                redundant_correlated_features = {i[0]: p for i, p in pairs.items() if p > correlated_features_th}
                _log.warning(f"drop redundant features: {redundant_correlated_features}")

                features_and_labels = PostProcessedFeaturesAndLabels.from_features_and_labels(
                    features_and_labels,
                    feature_post_processor=lambda df: df.drop(redundant_correlated_features.keys(), axis=1)
                )

            # estimate model type and sample properties
            is_classification = 'float' not in (str(features_and_labels.label_type))
            nr_samples = len(self.df)

            if is_classification:
                nr_classes = len(ext.labels.value_counts())
            else:
                nr_classes = max(len(self.df) / 3, 100)

            # estimate grid search parameters
            grid = {
                "estimator__n_estimators": sp_randint(10, 500),
                "estimator__max_depth": [2, None],
                "estimator__min_samples_split": sp_randint(2, nr_samples / nr_classes),
                "estimator__min_samples_leaf": sp_randint(2, nr_samples / nr_classes),
                "estimator__bootstrap": [True, False],
                "estimator__criterion": ["gini", "entropy"] if is_classification else ["mse", "mae"]
            }

            # build model
            cross_validation = TimeSeriesSplit if is_time_series else StratifiedKFold if is_classification else KFold
            estimator = RandomForestClassifier() if is_classification else RandomForestRegressor()
            selector = RFECV(estimator, step=1, cv=cross_validation(rfecv_splits), min_features_to_select=min_features_to_select)
            skm = RandomizedSearchCV(selector, param_distributions=grid, cv=cross_validation(forest_splits), n_jobs=-1)

            # fit model
            fit = m.fit(
                SkModel(skm, features_and_labels=features_and_labels, summary_provider=FeatureSelectionSummary),
                FittingParameter(splitter=training_data_splitter)
            )

        # we hide the loss plot from this summary
        return fit.with_hidden_loss_plot()

    def fit(self,
            model_provider: Callable[[], MlModel],
            fitting_parameter: FittingParameter = FittingParameter(),
            verbose: int = 0,
            callbacks: Union[Callable, List[Callable]] = None,
            fail_silent: bool = False,
            **kwargs
            ) -> Fit:
        df = self.df
        trails = None
        model = model_provider()
        kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)

        def fit_submodel(df, model, **kwargs):
            return model.fit(df, **kwargs)

        typemap_fitting = {SubModelFeature: fit_submodel, **self._type_mapping}
        frames: FeaturesWithLabels = model.features_and_labels(
            df, extract_feature_labels_weights, type_map=typemap_fitting, fitting_parameter=fitting_parameter,
            verbose=verbose, **kwargs
        )

        start_performance_count = perf_counter()
        _log.info("create model")

        df_train_prediction, df_test_prediction = model.fit(
            XYWeight(frames.features, frames.labels, frames.sample_weights),
            fitting_parameter,
            verbose,
            callbacks,
            **kwargs
        )

        _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")

        # assemble result objects
        # get training and test data tuples of the provided frames.
        # NOTE due to event boosting there might be duplicate events in the test data which we need to filter away
        ext_frames = frames.targets, frames.labels, frames.gross_loss, frames.sample_weights, frames.features
        df_train = assemble_result_frame(df_train_prediction, *ext_frames)
        df_test = assemble_result_frame(df_test_prediction[~df_test_prediction.index.duplicated()], *ext_frames)

        # update model properties and return the fit
        model.features_and_labels.set_min_required_samples(frames.features_with_required_samples.min_required_samples)
        model.features_and_labels._kwargs = {k: a for k, a in kwargs.items() if k in model.features_and_labels.kwargs}

        def assemble_fit():
            return Fit(
                model,
                model.summary_provider(df_train, model, is_test=False, **kwargs),
                model.summary_provider(df_test, model, is_test=True, **kwargs),
                trails,
                **kwargs
            )

        return call_silent(assemble_fit, lambda e: FitException(e, model)) if fail_silent else assemble_fit()

    def backtest(self,
                 model: MlModel,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = None,
                 tail: int = None,
                 **kwargs) -> Summary:

        min_required_samples = model.features_and_labels.min_required_samples
        df = self.df

        if tail is not None:
            if min_required_samples is not None:
                # just use the tail for feature engineering
                df = df[-(abs(tail) + (min_required_samples - 1)):]
            else:
                _log.warning("could not determine the minimum required data from the model")

        kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
        typemap_pred = {SubModelFeature: lambda df, model, **kwargs: model.predict(df, **kwargs), **self._type_mapping}
        frames: FeaturesWithLabels = model.features_and_labels(
            df, extract_feature_labels_weights, type_map=typemap_pred, **kwargs
        )

        predictions = model.predict(frames.features, **kwargs)
        df_backtest = assemble_result_frame(predictions, frames.targets, frames.labels, frames.gross_loss,
                                            frames.sample_weights, frames.features)

        return call_callable_dynamic_args(summary_provider or model.summary_provider, df_backtest, model, **kwargs)

    def predict(self,
                model: MlModel,
                tail: int = None,
                samples: int = 1,
                forecast_provider: Callable[[Typing.PatchedDataFrame], Forecast] = None,
                **kwargs) -> Union[Typing.PatchedDataFrame, Forecast]:
        min_required_samples = model.features_and_labels.min_required_samples
        df = self.df

        if tail is not None:
            if min_required_samples is not None:
                # just use the tail for feature engineering
                df = df[-(abs(tail) + (min_required_samples - 1)):]
            else:
                _log.warning("could not determine the minimum required data from the model")

        kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
        typemap_pred = {SubModelFeature: lambda df, model, **kwargs: model.predict(df, **kwargs), **self._type_mapping}
        frames: FeaturesWithTargets = model.features_and_labels(
            df, extract_features, type_map=typemap_pred, **kwargs
        )

        predictions = call_callable_dynamic_args(
            model.predict,
            features=frames.features, targets=frames.targets, latent=frames.latent, samples=samples, df=df,
            **kwargs
        )

        fc_provider = forecast_provider or model.forecast_provider
        res_df = assemble_result_frame(predictions, frames.targets, None, None, None, frames.features)

        return res_df if fc_provider is None else call_callable_dynamic_args(fc_provider, res_df, **kwargs)

    def __call__(self, file_name=None):
        from .model_context import ModelContext
        return ModelContext(self.df, file_name=file_name)
