from typing import Callable, Tuple, Dict, Union, Iterable, List

from pandas_ml_common import Typing, naive_splitter
from pandas_ml_common.utils import has_indexed_columns, merge_kwargs
from pandas_ml_utils.ml.data.analysis import feature_selection
from pandas_ml_utils.ml.data.analysis.plot_features import plot_features
from pandas_ml_utils.ml.data.extraction import extract_feature_labels_weights
from pandas_ml_utils.ml.data.extraction.features_and_labels_definition import FeaturesAndLabels
from pandas_ml_utils.ml.fitting import fit, backtest, predict, Fit
from pandas_ml_utils.ml.model import Model as MlModel
from pandas_ml_utils.ml.summary import Summary
from .model_context import ModelContext
from ..ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels


class DfModelPatch(object):

    def __init__(self, df: Typing.PatchedDataFrame):
        self.df = df

    def feature_selection(self,
                          features_and_labels: FeaturesAndLabels,
                          top_features: int = 5,
                          correlation_threshold: float = 0.5,
                          minimum_features: int = 1,
                          lags: Iterable[int] = range(100),
                          show_plots: bool = True,
                          figsize: Tuple[int, int] = (12, 10),
                          **kwargs):
        # extract pandas objects
        kwargs = merge_kwargs(features_and_labels.kwargs, kwargs)
        fl: FeaturesWithLabels = extract_feature_labels_weights(self.df, features_and_labels, **kwargs)

        # try to estimate good features
        return feature_selection(
            fl.features_with_required_samples.features,
            fl.labels,
            top_features,
            correlation_threshold,
            minimum_features,
            lags,
            show_plots,
            figsize
        )

    def plot_features(self, data: Union[FeaturesAndLabels, MlModel]):
        fnl = data.features_and_labels if isinstance(data, MlModel) else data
        fl: FeaturesWithLabels = self.df._.extract(fnl)

        return plot_features(
            fl.features_with_required_samples.features.join(fl.labels),
            fl.labels.columns[0] if has_indexed_columns(fl.labels) else fl.labels.name
        )

    def fit(self,
            model_provider: Callable[[], MlModel],
            training_data_splitter: Callable[[Typing.PdIndex], Tuple[Typing.PdIndex, Typing.PdIndex]] = naive_splitter(),
            training_samples_filter: Union['BaseCrossValidator', Tuple[int, Callable[[Typing.PatchedSeries], bool]]] = None,
            cross_validation: Tuple[int, Callable[[Typing.PdIndex], Tuple[List[int], List[int]]]] = None,
            hyper_parameter_space: Dict = None,
            **kwargs
            ) -> Fit:
        return fit(
            self.df,
            model_provider,
            training_data_splitter,
            training_samples_filter,
            cross_validation,
            hyper_parameter_space,
            **kwargs
        )

    def backtest(self,
                 model: MlModel,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = None,
                 **kwargs) -> Summary:
        return backtest(self.df, model, summary_provider, **kwargs)

    def predict(self,
                model: MlModel,
                tail: int = None,
                samples: int = 1,
                **kwargs) -> Typing.PatchedDataFrame:
        return predict(self.df, model, tail=tail, samples=samples, **kwargs)

    def __call__(self, file_name=None):
        return ModelContext(self.df, file_name=file_name)
