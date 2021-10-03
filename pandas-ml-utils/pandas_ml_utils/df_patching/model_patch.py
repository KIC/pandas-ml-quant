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
