from typing import Callable, Tuple, Dict

import numpy as np
import pandas as pd

from pandas_ml_utils.ml.fitting import fit, backtest, predict, Fit
from pandas_ml_utils.ml.model import Model as MlModel
from pandas_ml_utils.ml.summary import Summary


class Model(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fit(self,
            model_provider: Callable[[], MlModel],
            test_size: float = 0.4,
            youngest_size: float = None,
            cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
            test_validate_split_seed = 42,
            hyper_parameter_space: Dict = None
            ) -> Fit:
        return fit(self.df, model_provider, test_size, youngest_size, cross_validation, test_validate_split_seed, hyper_parameter_space)

    def backtest(self,
                 model: MlModel,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary) -> Summary:
        return backtest(self.df, model, summary_provider)

    def predict(self,
                model: MlModel,
                tail: int = None) -> pd.DataFrame:
        return predict(self.df, model, tail)

