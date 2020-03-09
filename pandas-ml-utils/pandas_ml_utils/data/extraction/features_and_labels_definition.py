import inspect
import logging
from copy import deepcopy
from typing import List, Callable, Iterable, Dict, Type, Tuple, Union, Any


_log = logging.getLogger(__name__)
_LABELS = Union[List[str], TargetLabelEncoder, Dict[str, Union[List[str], TargetLabelEncoder]]]
_LABELS = Union[_LABELS, Callable[[Any], _LABELS]]

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
                 features: List[str],
                 labels: _LABELS,
                 label_type: Type = None,
                 sample_weights: Union[Dict[str, str], str] = None,
                 gross_loss: Callable[[str, pd.DataFrame], Union[pd.Series, pd.DataFrame]] = None,
                 targets: Callable[[str, pd.DataFrame], Union[pd.Series, pd.DataFrame]] = None,
                 feature_lags: Iterable[int] = None,
                 feature_rescaling: Dict[Tuple[str, ...], Tuple[int, ...]] = None,  # TODO lets provide a rescaler ..
                 lag_smoothing: Dict[int, Callable[[pd.Series], pd.Series]] = None,
                 pre_processor: Callable[[pd.DataFrame, Dict], pd.DataFrame] = lambda x: x,
                 **kwargs):
        """
        :param features: a list of column names which are used as features for your model
        :param labels: as list of column names which are uses as labels for your model. you can specify one ore more
                       named targets for a set of labels by providing a dict. This is useful if you want to train a
                       :class:`.MultiModel` or if you want to provide extra information about the label. i.e. you
                       want to classify whether a stock price is below or above average and you want to provide what
                       the average was. It is also possible to provide a Callable[[df, ...magic], labels] which returns
                       the expected .data structure.
        :param label_type: whether to treat a label as int, float, bool
        :param sample_weights: sample weights get passed to the model.fit function. In keras for example this can be
                               used for imbalanced classes
        :param gross_loss: expects a callable[[df, target, ...magic], df] which receives the source .data frame and a
                           target (or None) and should return a series or .data frame. Let's say you want to classify
                           whether a printer is jamming the next page or not. Halting and servicing the printer costs
                           5'000 while a jam costs 15'000. Your target will be 0 or empty but your gross loss will be
                           -5000 for all your type II errors and -15'000 for all your type I errors in case of miss-
                           classification. Another example would be if you want to classify whether a stock price is
                           above (buy) the current price or not (do nothing). Your target is the today's price and your
                           loss is tomorrows price minus today's price.
        :param targets: expects a callable[[df, targets, ...magic], df] which receives the source .data frame and a
                        target (or None) and should return a series or .data frame. In case of multiple targets the
                        series names need to be unique!
        :param feature_lags: an iterable of integers specifying the lags of an AR model i.e. [1] for AR(1)
                             if the un-lagged feature is needed as well provide also lag of 0 like range(1)
        :param feature_rescaling: this allows to rescale features.
                                  in a dict we can define a tuple of column names and a target range
        :param lag_smoothing: very long lags in an AR model can be a bit fuzzy, it is possible to smooth lags i.e. by
                              using moving averages. the key is the lag length at which a smoothing function starts to
                              be applied
        :param pre_processor: provide a callable[[df, ...magic], df] returning an eventually augmented .data frame from
                              a given source .data frame and self.kwargs. This is useful if you have i.e. .data cleaning
                              tasks. This way you can apply the model directly on the raw .data.
        :param kwargs: maybe you want to pass some extra parameters to a callable you have provided
        """
        self._features = features
        self._labels = labels
        self._weights = sample_weights
        self._targets = targets
        self._gross_loss = gross_loss
        self.label_type = label_type
        self.feature_lags = [lag for lag in feature_lags] if feature_lags is not None else None
        self.feature_rescaling = feature_rescaling
        self.lag_smoothing = lag_smoothing
        self.len_feature_lags = sum(1 for _ in self.feature_lags) if self.feature_lags is not None else 1
        self.expanded_feature_length = len(features) * self.len_feature_lags if feature_lags is not None else len(features)
        self._min_required_samples = None
        self.pre_processor = pre_processor
        self.kwargs = kwargs
        _log.info(f'number of features, lags and total: {self.len_features()}')

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def weights(self):
        return self._weights

    @property
    def targets(self):
        return self._targets

    @property
    def gross_loss(self):
        return self._gross_loss

    @property
    def min_required_samples(self):
        return self._min_required_samples

    @property
    def shape(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Returns the shape of features and labels how they get passed to the :class:`.Model`. If laging is used, then
        the features shape is in Keras RNN form.

        :return: a tuple of (features.shape, labels.shape)
        """
        if self.feature_lags is not None:
            return (len(self.feature_lags), len(self.features)), (self.len_labels(), )
        else:
            return (len(self.features), ), (self.len_labels(), )

    def len_features(self) -> Tuple[int, ...]:
        """
        Returns the length of the defined features, the number of lags used and the total number of all features * lags

        :return: tuple of (#features, #lags, #features * #lags)
        """

        return len(self.features), self.len_feature_lags, self.expanded_feature_length

    def len_labels(self) -> int:
        """
        Returns the number of labels

        :return:  number of labels
        """
        return len(self.labels)

    def with_labels(self, labels: _LABELS):
        copy = deepcopy(self)
        copy._labels = labels
        return copy

    def with_kwargs(self, **kwargs):
        copy = deepcopy(self)
        copy.kwargs = join_kwargs(self.kwargs, kwargs)
        return copy

    def __call__(self, extractor: callable, *args, **kwargs):
        # FIXME the goal is to perform something like df.ml.extract(self: FeaturesAndLabels) where we define a default
        # extractor
        pass

    def __repr__(self):
        return f'FeaturesAndLabels({self.features},{self.labels},{self.targets},' \
               f'{self.feature_lags},{self.feature_rescaling}{self.lag_smoothing}) ' \
               f'#{len(self.features)} ' \
               f'features expand to {self.expanded_feature_length}'

    def __hash__(self):
        return hash(self.__id__())

    def __eq__(self, other):
        return self.__id__() == other.__id__()

    def __id__(self):
        import dill  # only import if really needed
        smoothers = ""

        if self.lag_smoothing is not None:
            smoothers = {feature: inspect.getsource(smoother) for feature, smoother in self.lag_smoothing.items()}

        return f'{self.features},{self.labels},{self.label_type},{self.targets},{dill.dumps(self.feature_lags)},{self.feature_rescaling},{smoothers}'

    def __str__(self):
        return self.__repr__()


