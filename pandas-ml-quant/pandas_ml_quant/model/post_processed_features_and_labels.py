from pandas_ml_utils import FeaturesAndLabels, Typing, Callable, List
from pandas_ml_utils.ml.data.extraction import extract_with_post_processor


class PostProcessedFeaturesAndLabels(FeaturesAndLabels):

    def __init__(self,
                 features: Typing._Selector,
                 feature_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]],
                 labels: Typing._Selector,
                 labels_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 sample_weights: Typing._Selector = None,
                 gross_loss: Typing._Selector = None,
                 targets: Typing._Selector = None,
                 label_type=None,
                 **kwargs):
        super().__init__(
            PostProcessedFeaturesAndLabels.post_process(features, feature_post_processor),
            PostProcessedFeaturesAndLabels.post_process(labels, labels_post_processor),
            sample_weights,
            gross_loss,
            targets,
            label_type,
            **kwargs
        )

    @staticmethod
    def post_process(selectors, post_processors):
        # early exit if we actually do not post process
        if post_processors is None:
            return selectors

        # make post processors iterable and exceute one after the other
        pps = post_processors if isinstance(post_processors, list) else [post_processors]
        previous = selectors

        for pp in pps:
            previous = extract_with_post_processor(previous, pp)

        return previous