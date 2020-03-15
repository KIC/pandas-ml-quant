from typing import Callable, Tuple, Dict, Union, List, Iterable

import numpy as np
import pandas as pd

from pandas_ml_common import get_pandas_object
from pandas_ml_utils.ml.data.analysis import feature_selection
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels
from pandas_ml_utils.ml.fitting import fit, backtest, predict, Fit
from pandas_ml_utils.ml.model import Model as MlModel
from pandas_ml_utils.ml.summary import Summary


class Model(object):

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def feature_selection(self,
                          features_and_labels: FeaturesAndLabels,
                          top_features: int = 5,
                          correlation_threshold: float = 0.5,
                          minimum_features: int = 1,
                          lags: Iterable[int] = range(100),
                          show_plots: bool = True,
                          figsize: Tuple[int, int] = (12, 10)):
        # extract pandas objects
        features = get_pandas_object(self.df, features_and_labels.features)
        label = get_pandas_object(self.df, features_and_labels.labels)

        # try to estimate good features
        return feature_selection(features, label, top_features, correlation_threshold, minimum_features,
                                 lags, show_plots, figsize)

    def fit(self,
            model_provider: Callable[[], MlModel],
            test_size: float = 0.4,
            youngest_size: float = None,
            cross_validation: Tuple[int, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
            test_validate_split_seed = 42,
            hyper_parameter_space: Dict = None,
            **kwargs
            ) -> Fit:
        return fit(self.df, model_provider, test_size, youngest_size, cross_validation, test_validate_split_seed,
                   hyper_parameter_space, **kwargs)

    def backtest(self,
                 model: MlModel,
                 summary_provider: Callable[[pd.DataFrame], Summary] = Summary,
                 **kwargs) -> Summary:
        return backtest(self.df, model, summary_provider, **kwargs)

    def predict(self,
                model: MlModel,
                tail: int = None,
                **kwargs) -> pd.DataFrame:
        return predict(self.df, model, tail, **kwargs)

