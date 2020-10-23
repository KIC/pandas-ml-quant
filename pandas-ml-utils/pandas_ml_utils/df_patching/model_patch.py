import logging
from time import perf_counter
from typing import Callable, Tuple, Dict, Union, List

from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold

from pandas_ml_common import Typing, naive_splitter, Sampler, XYWeight
from pandas_ml_common.utils import has_indexed_columns, get_correlation_pairs, merge_kwargs
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels, \
    PostProcessedFeaturesAndLabels
from pandas_ml_utils.ml.data.reconstruction import assemble_result_frame
from pandas_ml_utils.ml.fitting import Fit, FitException
from pandas_ml_utils.ml.model import Model as MlModel, SkModel
from pandas_ml_utils.ml.summary import Summary
from .model_context import ModelContext
from ..ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels, extract, FeaturesWithTargets, \
    extract_features, extract_feature_labels_weights
from ..ml.summary.feature_selection_summary import FeatureSelectionSummary

_log = logging.getLogger(__name__)


class DfModelPatch(object):

    def __init__(self, df: Typing.PatchedDataFrame):
        self.df = df

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
                _, pairs = get_correlation_pairs(ext.features_with_required_samples.features)
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
                training_data_splitter=training_data_splitter,
            )

        # we hide the loss plot from this summary
        return fit.with_hidden_loss_plot()

    # Obsolete ... might be part of the report of the feature analysis/selection section
    def plot_features(self, data: Union[FeaturesAndLabels, MlModel]):
        # FIXME this is the only seaborn dependency left, get rid of it if possible
        fnl = data.features_and_labels if isinstance(data, MlModel) else data
        fl: FeaturesWithLabels = self.df._.extract(fnl)

        def plot_features(joined_features_andLabels_df, label_column):
            import seaborn as sns

            # fixme if labels are contonious, we need to bin them
            # fixme if one hot encoded label column use np.argmax
            return sns.pairplot(joined_features_andLabels_df, hue=label_column)

        return plot_features(
            fl.features_with_required_samples.features.join(fl.labels),
            fl.labels.columns[0] if has_indexed_columns(fl.labels) else fl.labels.name
        )

    def fit(self,
            model_provider: Callable[[], MlModel],
            splitter: Callable[[Typing.PdIndex], Tuple[Typing.PdIndex, Typing.PdIndex]] = naive_splitter(),
            filter: Union['BaseCrossValidator', Tuple[int, Callable[[Typing.PatchedSeries], bool]]] = None,
            cross_validation: Tuple[int, Callable[[Typing.PdIndex], Tuple[List[int], List[int]]]] = None,
            epochs: int = 1,
            batch_size: int = None,
            fold_epochs: int = 1,
            hyper_parameter_space: Dict = None,
            silent: bool = False,
            **kwargs
            ) -> Fit:
        df = self.df
        trails = None
        model = model_provider()
        kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
        frames: FeaturesWithLabels = extract(model.features_and_labels, df, extract_feature_labels_weights, **kwargs)

        start_performance_count = perf_counter()
        _log.info("create model")

        df_train_prediction, df_test_prediction = model.fit(
            Sampler(
                XYWeight(frames.features_with_required_samples.features, frames.labels, frames.sample_weights),
                splitter=splitter,
                filter=filter,
                cross_validation=cross_validation,
                epochs=epochs,
                fold_epochs=fold_epochs,
                batch_size=batch_size
            ),
            **kwargs
        )

        _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")

        # assemble result objects
        try:
            # get training and test data tuples of the provided frames
            ext_frames = frames.targets, frames.labels, frames.gross_loss, frames.sample_weights, frames.features_with_required_samples.features
            df_train = assemble_result_frame(df_train_prediction, *ext_frames)
            df_test = assemble_result_frame(df_test_prediction, *ext_frames)

            # update model properties and return the fit
            model.features_and_labels.set_min_required_samples(frames.features_with_required_samples.min_required_samples)
            model.features_and_labels._kwargs = {k: a for k, a in kwargs.items() if k in model.features_and_labels.kwargs}

            return Fit(
                model,
                model.summary_provider(df_train, model, is_test=False, **kwargs),
                model.summary_provider(df_test, model, is_test=True, **kwargs),
                trails,
                **kwargs
            )
        except Exception as e:
            fex = FitException(e, model)
            if not silent:
                raise fex
            else:
                return fex

    def backtest(self,
                 model: MlModel,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = None,
                 **kwargs) -> Summary:
        df = self.df
        kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
        frames: FeaturesWithLabels = extract(model.features_and_labels, df, extract_feature_labels_weights, **kwargs)

        predictions = model.predict(frames.features_with_required_samples.features, **kwargs)
        df_backtest = assemble_result_frame(predictions, frames.targets, frames.labels, frames.gross_loss,
                                            frames.sample_weights, frames.features_with_required_samples.features)

        return (summary_provider or model.summary_provider)(df_backtest, model, **kwargs)

    def predict(self,
                model: MlModel,
                tail: int = None,
                samples: int = 1,
                **kwargs) -> Typing.PatchedDataFrame:
        min_required_samples = model.features_and_labels.min_required_samples
        df = self.df

        if tail is not None:
            if min_required_samples is not None:
                # just use the tail for feature engineering
                df = df[-(abs(tail) + (min_required_samples - 1)):]
            else:
                _log.warning("could not determine the minimum required data from the model")

        kwargs = merge_kwargs(model.features_and_labels.kwargs, model.kwargs, kwargs)
        frames: FeaturesWithTargets = extract(model.features_and_labels, df, extract_features, **kwargs)

        # features, labels, targets, weights, gross_loss, latent,
        predictions = model.predict(frames.features, samples, **kwargs)
        return assemble_result_frame(predictions, frames.targets, None, None, None, frames.features)

    def __call__(self, file_name=None):
        return ModelContext(self.df, file_name=file_name)
