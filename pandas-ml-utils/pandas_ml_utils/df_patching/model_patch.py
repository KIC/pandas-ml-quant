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
            **kwargs
            ) -> Fit:
        df = self.df

        start_performance_count = perf_counter()
        _log.info("create model")

        fit = model.fit_to_df(
            df,
            fitting_parameter,
            verbose,
            callbacks,
            **kwargs
        )

        _log.info(f"fitting model done in {perf_counter() - start_performance_count: .2f} sec!")

        # FIXME move directly to ModelContext
        return fit

    def backtest(self,
                 model: Fittable,
                 summary_provider: Callable[[MlTypes.PatchedDataFrame], Summary] = None,
                 tail: int = None,
                 **kwargs) -> Summary:
        # FIXME move directly to ModelContext
        return model.forecast(self.df, tail, 1, summary_provider or model.summary_provider, True, **kwargs)

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
