import inspect
from copy import deepcopy
from typing import List, Callable, Tuple, Union, Any, TypeVar

import pandas as pd

from pandas_ml_common.utils import as_list
from pandas_ml_common.utils.callable_utils import call_callable_dynamic_args
from pandas_ml_common import Typing, get_pandas_object

T = TypeVar('T', str, List, Callable[[Any], Union[pd.DataFrame, pd.Series]])


# This class should be able to be pickled and unpickled without risk of change between versions
# This means business logic need to be kept outside of this class!
class FeaturesAndLabels(object):
    """
    *FeaturesAndLabels* is the main object used to hold the context of your problem. Here you define which columns
    of your `DataFrame` is a feature, a label or a target. This class also provides some functionality to generate
    autoregressive features. By default lagging features results in an RNN shaped 3D array (in the format of keras
    RNN layers input format).
    """

    def __init__(self,
                 features: Union[Typing._Selector, Tuple[Typing._Selector]],
                 labels: Typing._Selector = [],
                 latent: Typing._Selector = None,
                 latent_names: List[str] = None,
                 sample_weights: Typing._Selector = None,
                 gross_loss: Typing._Selector = None,
                 targets: Typing._Selector = None,
                 label_type = None,
                 **kwargs):
        """
        :param features: a list of column names which are used as features for your model
        :param labels: as list of column names which are uses as labels for your model. you can specify one ore more
            named targets for a set of labels by providing a dict. This is useful if you want to train a
            :class:`.MultiModel` or if you want to provide extra information about the label. i.e. you
            want to classify whether a stock price is below or above average and you want to provide what
            the average was. It is also possible to provide a Callable[[df, ...magic], labels] which returns
            the expected .data structure.
        :param latent: a list of column names which are used as labels / features of an auto encoder model
        :param sample_weights: sample weights get passed to the model.fit function. In keras for example this can be
             used for imbalanced classes
        :param gross_loss: expects a callable[[df, target, ...magic], df] which receives the source .data frame and a
            target (or None) and should return a series or .data frame. Let's say you want to classify whether a
            printer is jamming the next page or not. Halting and servicing the printer costs 5'000 while a jam costs
            15'000. Your target will be 0 or empty but your gross loss will be -5000 for all your type II errors and
            -15'000 for all your type I errors in case of miss-classification. Another example would be if you want to
            classify whether a stock price is above (buy) the current price or not (do nothing). Your target is the
            today's price and your loss is tomorrows price minus today's price.
        :param targets: expects a callable[[df, targets, ...magic], df] which receives the source .data frame and a
            target (or None) and should return a series or .data frame. In case of multiple targets the series names
            need to be unique!
        :param min_required_samples: in case you only want to do one prediction and have some feature engineering
            you might need some minimum amount of samples to engineer your features. This can be None but it increases
            performance if you provide it.
        :param label_type: cast label to its type after all nans got dropped
        :param kwargs: maybe you want to pass some extra parameters to a callable you have provided
        """

        self._features = features
        self._labels = labels
        self._latent = latent
        self._label_type = label_type
        self._sample_weights = sample_weights
        self._targets = targets
        self._gross_loss = gross_loss
        self._kwargs = kwargs

        if label_type == 'regression':
            self._label_type = float
        elif label_type == 'classification':
            self._label_type = int

        if latent is not None:
            self._latent_names = latent_names if latent_names is not None else [str(l) for l in latent]
        else:
            self._latent_names = None

        # set after fit
        self._min_required_samples = None

    def set_min_required_samples(self, min_required: int):
        self._min_required_samples = min_required

        return self

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def latent(self):
        return self._latent

    @property
    def latent_names(self):
        return self._latent_names

    @property
    def label_type(self):
        return self._label_type

    @property
    def sample_weights(self):
        return self._sample_weights

    @property
    def targets(self):
        return self._targets

    @property
    def gross_loss(self):
        return self._gross_loss

    @property
    def min_required_samples(self) -> Union[int, Callable[[Any], int]]:
        return self._min_required_samples

    @property
    def kwargs(self):
        return self._kwargs

    def with_labels(self, labels: Union[str, List[T], Callable[[Any], Union[pd.DataFrame, pd.Series]]]):
        copy = deepcopy(self)
        copy._labels = labels
        return copy

    def with_sample_weights(self, sample_weights: Union[str, List[T], Callable[[Any], Union[pd.DataFrame, pd.Series]]]):
        copy = deepcopy(self)
        copy._sample_weights = sample_weights
        return copy

    def with_gross_loss(self, gross_loss: Union[str, List[T], Callable[[Any], Union[pd.DataFrame, pd.Series]]]):
        copy = deepcopy(self)
        copy._gross_loss = gross_loss
        return copy

    def with_kwargs(self, **kwargs):
        copy = deepcopy(self)
        copy.kwargs = {**self.kwargs, **kwargs}
        return copy

    def __call__(self,
                 df: pd.DataFrame,
                 extractor: callable = None,
                 *args,
                 **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        # Basic concept:
        #  we call an extractor(df, **{**self.kwargs, **kwargs})
        #  this extractor uses 'get_pandas_object' which itself can handle lambdas with dependecies
        #  injected from available kwargs
        if extractor is None or extractor in ['training', 'train']:
            from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import extract_feature_labels_weights
            extractor = extract_feature_labels_weights
        elif extractor in ['eval', 'evaluate', 'predict']:
            from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import extract_features
            extractor = extract_features

        return call_callable_dynamic_args(extractor, df, self, **{**self.kwargs, **kwargs})

    def __hash__(self):
        return hash(self.__id__())

    def __id__(self):
        return self.__repr__()

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.__id__() == other.__id__() if isinstance(other, FeaturesAndLabels) else False

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

                return [insp(p) if callable(p) else p if isinstance(p, (bool, int, float, str)) else repr(p) for p in params]

        return f'FeaturesAndLabels(' \
               f'\t{source(self.features)}, ' \
               f'\t{source(self.labels)}, ' \
               f'\t{source(self.latent)}, ' \
               f'\t{source(self.sample_weights)}, ' \
               f'\t{source(self.gross_loss)}, ' \
               f'\t{source(self.targets)}, ' \
               f'\t{self.label_type}, ' \
               f'\t{self.kwargs}' \
               f')'


PostProcessorType = Union[
    List[Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]],
    Callable[[Typing.PatchedDataFrame], Typing.PatchedDataFrame]
]


class PostProcessedFeaturesAndLabels(FeaturesAndLabels):

    @staticmethod
    def from_features_and_labels(
            features_and_labels: FeaturesAndLabels,
            feature_post_processor: Union[PostProcessorType, Tuple[PostProcessorType]] = None,
            labels_post_processor: PostProcessorType = None,
            latent_post_processor: PostProcessorType = None,
            sample_weights_post_processor: PostProcessorType = None,
            gross_loss_post_processor: PostProcessorType = None,
            targets_post_processor: PostProcessorType = None,
            **kwargs
    ):
        return PostProcessedFeaturesAndLabels(
            features=features_and_labels.features,
            feature_post_processor=feature_post_processor,
            labels=features_and_labels.labels,
            labels_post_processor=labels_post_processor,
            latent=features_and_labels.latent,
            latent_names=features_and_labels.latent_names,
            latent_post_processor=latent_post_processor,
            sample_weights=features_and_labels.sample_weights,
            sample_weights_post_processor=sample_weights_post_processor,
            gross_loss=features_and_labels.gross_loss,
            gross_loss_post_processor=gross_loss_post_processor,
            targets=features_and_labels.targets,
            targets_post_processor=targets_post_processor,
            label_type=features_and_labels.label_type,
            **{**features_and_labels._kwargs, **kwargs}
        )

    def __init__(self,
                 features: Typing._Selector,
                 feature_post_processor: Union[PostProcessorType, Tuple[PostProcessorType]] = None,
                 labels: Typing._Selector = [],
                 labels_post_processor: PostProcessorType = None,
                 latent: Typing._Selector = None,
                 latent_names: List[str] = None,
                 latent_post_processor: PostProcessorType = None,
                 sample_weights: Typing._Selector = None,
                 sample_weights_post_processor: PostProcessorType = None,
                 gross_loss: Typing._Selector = None,
                 gross_loss_post_processor: PostProcessorType = None,
                 targets: Typing._Selector = None,
                 targets_post_processor: PostProcessorType = None,
                 label_type=None,
                 **kwargs):
        super().__init__(
            features=PostProcessedFeaturesAndLabels.post_process(features, as_list(feature_post_processor)),
            labels=PostProcessedFeaturesAndLabels.post_process(labels, as_list(labels_post_processor)),
            latent=PostProcessedFeaturesAndLabels.post_process(latent, as_list(latent_post_processor)),
            latent_names=latent_names,
            sample_weights=PostProcessedFeaturesAndLabels.post_process(sample_weights, as_list(sample_weights_post_processor)),
            gross_loss=PostProcessedFeaturesAndLabels.post_process(gross_loss, as_list(gross_loss_post_processor)),
            targets=PostProcessedFeaturesAndLabels.post_process(targets, as_list(targets_post_processor)),
            label_type=label_type,
            **kwargs
        )

        self._raw = [
            [features, feature_post_processor],
            [labels, labels_post_processor],
            [latent, latent_post_processor],
            [sample_weights, sample_weights_post_processor],
            [gross_loss, gross_loss_post_processor],
            [targets, targets_post_processor],
        ]

    @staticmethod
    def post_process(selectors, post_processors):
        # early exit if we actually do not post process
        if post_processors is None:
            return selectors

        # if we define multiple feature frames we need to post process every single one of them
        if isinstance(selectors, tuple) and len(selectors) > 1 and isinstance(selectors[0], (tuple, list)):
            return tuple([PostProcessedFeaturesAndLabels.post_process(selectors[i], post_processors[i]) for i in range(len(selectors))])

        def extract_with_post_processor(selectors, postprocessor: Callable):
            def extractor(df, **kwargs):
                return get_pandas_object(get_pandas_object(df, selectors, **kwargs), postprocessor, **kwargs)

            return extractor

        previous = selectors
        for pp in post_processors:
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
                        (params if isinstance(params, (List, Tuple)) else [params])]

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
               f'\t{source(self._raw[5][0])}, ' \
               f'\t{source(self._raw[5][1])}, ' \
               f'\t{self.label_type}, ' \
               f'\t{self.kwargs}' \
               f')'
