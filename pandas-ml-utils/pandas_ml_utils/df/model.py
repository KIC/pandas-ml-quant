from typing import Callable, Tuple, Dict, Union, Iterable

from pandas_ml_common import get_pandas_object, Typing
from pandas_ml_common.utils import has_indexed_columns
from pandas_ml_utils.ml.data.analysis import feature_selection
from pandas_ml_utils.ml.data.analysis.plot_features import plot_features
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels
from pandas_ml_utils.ml.data.splitting import Splitter, RandomSplits
from pandas_ml_utils.ml.fitting import fit, backtest, predict, Fit
from pandas_ml_utils.ml.model import Model as MlModel
from pandas_ml_utils.ml.summary import Summary


class Model(object):

    def __init__(self, df: Typing.PatchedDataFrame):
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

    def plot_features(self, data: Union[FeaturesAndLabels, MlModel]):
        fnl = data.features_and_labels if isinstance(data, MlModel) else data
        (features, _), labels, _, _, _ = self.df._.extract(fnl)
        return plot_features(features.join(labels), labels.columns[0] if has_indexed_columns(labels) else labels.name)

    def fit(self,
            model_provider: Callable[[], MlModel],
            training_data_splitter: Splitter = RandomSplits(),
            hyper_parameter_space: Dict = None,
            **kwargs
            ) -> Fit:
        return fit(self.df, model_provider, training_data_splitter, hyper_parameter_space, **kwargs)

    def backtest(self,
                 model: MlModel,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs) -> Summary:
        return backtest(self.df, model, summary_provider, **kwargs)

    def predict(self,
                model: MlModel,
                tail: int = None,
                samples: int = 1,
                **kwargs) -> Typing.PatchedDataFrame:
        return predict(self.df, model, tail=tail, samples=samples, **kwargs)

