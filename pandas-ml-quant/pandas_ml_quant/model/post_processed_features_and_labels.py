import logging
from typing import List, Callable

from pandas_ml_common import Typing
from pandas_ml_utils import PostProcessedFeaturesAndLabels as PostProcessedFeaturesAndLabelsUtils

log = logging.getLogger(__name__)


class PostProcessedFeaturesAndLabels(PostProcessedFeaturesAndLabelsUtils):

    def __init__(self, features: Typing._Selector,
                 feature_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]],
                 labels: Typing._Selector = [],
                 labels_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 sample_weights: Typing._Selector = None, sample_weights_post_processor: List[
                Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 gross_loss: Typing._Selector = None,
                 gross_loss_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 targets: Typing._Selector = None,
                 targets_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 label_type=None, **kwargs):
        super().__init__(features, feature_post_processor, labels, labels_post_processor, sample_weights,
                         sample_weights_post_processor, gross_loss, gross_loss_post_processor, targets,
                         targets_post_processor, label_type, **kwargs)

        log.error("This class is only here for backward compatibility of saved Models, use: "
                  "`from pandas_ml_utils import PostProcessedFeaturesAndLabels` instead!!")