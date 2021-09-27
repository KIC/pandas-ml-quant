import logging
from time import perf_counter
from typing import Callable, Union, List

from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold, StratifiedKFold

from pandas_ml_common import MlTypes, FeaturesLabels, naive_splitter
from pandas_ml_common.utils import get_correlation_pairs, call_silent, call_callable_dynamic_args
from pandas_ml_utils.ml.data.reconstruction import assemble_result_frame
from pandas_ml_utils.ml.fitting import Fit, FitException, FittingParameter
from pandas_ml_utils.ml.forecast import Forecast
from pandas_ml_utils.ml.model import Fittable, AutoEncoderModel, FittableModel
from pandas_ml_utils.ml.summary import Summary
from ..ml.summary.feature_selection_summary import FeatureSelectionSummary

_log = logging.getLogger(__name__)


class DfModelPatch(object):

    def __init__(self, df: MlTypes.PatchedDataFrame):
        self.df = df

    def feature_selection(self,
                          features_and_labels: FeaturesLabels,
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
            model: Fittable,
            fitting_parameter: FittingParameter = FittingParameter(),
            verbose: int = 0,
            callbacks: Union[Callable, List[Callable]] = None,
            fail_silent: bool = False,
            **kwargs
            ) -> Fit:
        df = self.df
        trails = None

        start_performance_count = perf_counter()
        _log.info("create model")

        df_train, df_test = model.fit_to_df(
            df,
            fitting_parameter,
            verbose,
            callbacks,
            **kwargs
        )

        _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")

        def assemble_fit():
            return Fit(
                model,
                model.summary_provider(df_train, model, is_test=False, **kwargs),
                model.summary_provider(df_test, model, is_test=True, **kwargs),
                trails,
                **kwargs
            )

        # FIXME move directly to ModelContext
        return call_silent(assemble_fit, lambda e: FitException(e, model)) if fail_silent else assemble_fit()

    def backtest(self,
                 model: Fittable,
                 summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = None,
                 tail: int = None,
                 **kwargs) -> Summary:
        # FIXME move directly to ModelContext
        df_backtest = model.forecast(self.df, tail, 1, None, True, **kwargs)
        return call_callable_dynamic_args(summary_provider or model.summary_provider, df=df_backtest, model=model, **kwargs)

    def predict(self,
                model: Fittable,
                tail: int = None,
                samples: int = 1,
                forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                **kwargs) -> Union[MlTypes.PatchedDataFrame, Forecast]:
        # FIXME move directly to ModelContext
        return model.forecast(self.df, tail, samples, forecast_provider, False, **kwargs)

    def __call__(self, file_name=None):
        from .model_context import ModelContext
        return ModelContext(self.df, file_name=file_name)
