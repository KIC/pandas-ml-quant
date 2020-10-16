from typing import Callable, Tuple

from pandas_ml_common import Typing, Sampler
from pandas_ml_utils import Model, FeaturesAndLabels, Summary


class RollingModel(Model):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 summary_provider: Callable[[Typing.PatchedDataFrame], Summary] = Summary,
                 **kwargs):
        super().__init__(features_and_labels, summary_provider, **kwargs)
        self.past_predictions = None

    def _fit(self, sampler: Sampler, **kwargs) -> Tuple[Typing.PatchedDataFrame, Typing.PatchedDataFrame, Typing.PatchedDataFrame]:
        # we need to intercept the sampler, call predict and remember the result for each index
        pass

    def _predict(self, sampler: Sampler, **kwargs) -> Typing.PatchedDataFrame:
        pass



