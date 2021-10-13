import logging
from time import perf_counter
from typing import Callable, Union, List

from pandas_ml_common import MlTypes
from pandas_ml_common.utils import call_silent
from ..ml.fitting import Fit, FitException, FittingParameter
from ..ml.forecast import Forecast
from ..ml.summary import Summary
from ..ml.model import Fittable, Model

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
                self.df, df_train, df_test,
                model.summary_provider,
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
        return Summary.provide(
            summary_provider or model.summary_provider,
            df_backtest, model, self.df, **kwargs
        )

    def predict(self,
                model: Model,
                tail: int = None,
                samples: int = 1,
                forecast_provider: Callable[[MlTypes.PatchedDataFrame], Forecast] = None,
                **kwargs) -> Union[MlTypes.PatchedDataFrame, Forecast]:
        # FIXME move directly to ModelContext
        return model.forecast(self.df, tail, samples, forecast_provider, False, **kwargs)

    def __call__(self, file_name=None):
        from .model_context import ModelContext
        return ModelContext(self.df, file_name=file_name)
