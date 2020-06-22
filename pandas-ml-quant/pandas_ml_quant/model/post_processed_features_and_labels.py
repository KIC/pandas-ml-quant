import inspect

from pandas_ml_utils import FeaturesAndLabels, Typing, Callable, List
from pandas_ml_utils.ml.data.extraction import extract_with_post_processor


class PostProcessedFeaturesAndLabels(FeaturesAndLabels):

    def __init__(self,
                 features: Typing._Selector,
                 feature_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]],
                 labels: Typing._Selector,
                 labels_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 sample_weights: Typing._Selector = None,
                 sample_weights_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 gross_loss: Typing._Selector = None,
                 gross_loss_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 targets: Typing._Selector = None,
                 targets_post_processor: List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]] = None,
                 label_type=None,
                 **kwargs):
        super().__init__(
            PostProcessedFeaturesAndLabels.post_process(features, feature_post_processor),
            PostProcessedFeaturesAndLabels.post_process(labels, labels_post_processor),
            PostProcessedFeaturesAndLabels.post_process(sample_weights, sample_weights_post_processor),
            PostProcessedFeaturesAndLabels.post_process(gross_loss, gross_loss_post_processor),
            PostProcessedFeaturesAndLabels.post_process(targets, targets_post_processor),
            label_type,
            **kwargs
        )

        self._raw = [
            [features, feature_post_processor],
            [labels, labels_post_processor],
            [sample_weights, sample_weights_post_processor],
            [gross_loss, gross_loss_post_processor],
            [targets, targets_post_processor],
        ]

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

    def __repr__(self):
        def source(params):
            if params is None:
                return None
            else:
                def insp(p):
                    try:
                        return inspect.getsource(p)
                    except Exception as e:
                        return p

                return [insp(p) if callable(p) else p if isinstance(p, (bool, int, float, str)) else repr(p) for p in
                        params]

        return f'PostProcessedFeaturesAndLabels(' \
               f'\t{source(self._raw[0][0])}, ' \
               f'\t{source(self._raw[0][1])}, ' \
               f'\t{source(self._raw[1][0])}, ' \
               f'\t{source(self._raw[1][1])}, ' \
               f'\t{source(self._raw[2][0])}, ' \
               f'\t{source(self._raw[2][1])}, ' \
               f'\t{source(self._raw[3][0])}, ' \
               f'\t{source(self._raw[3][1])}, ' \
               f'\t{source(self._raw[4][0])}, ' \
               f'\t{source(self._raw[4][1])}, ' \
               f'\t{self.label_type}, ' \
               f'\t{self.kwargs}' \
               f')'
