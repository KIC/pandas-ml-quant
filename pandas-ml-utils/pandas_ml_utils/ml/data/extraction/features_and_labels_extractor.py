import logging
from typing import NamedTuple, List, Tuple

import numpy as np
import pandas as pd

from pandas_ml_common import get_pandas_object, Typing
from pandas_ml_common.decorator import MultiFrameDecorator
from pandas_ml_common.utils import intersection_of_index, loc_if_not_none, flatten_nested_list
from pandas_ml_common.utils.callable_utils import call_if_not_none

_log = logging.getLogger(__name__)


class FeaturesWithTargets(NamedTuple):
    features: pd.DataFrame
    targets: pd.DataFrame
    latent: pd.DataFrame


class FeaturesWithRequiredSamples(NamedTuple):
    features: pd.DataFrame
    min_required_samples: int
    nr_of_features: int


class FeaturesWithLabels(NamedTuple):
    features_with_required_samples: FeaturesWithRequiredSamples
    labels: pd.DataFrame
    latent: pd.DataFrame
    targets: pd.DataFrame
    sample_weights: pd.DataFrame
    gross_loss: pd.DataFrame

    @property
    def features(self):
        return self.features_with_required_samples.features


def extract_feature_labels_weights(
        df: Typing.PatchedDataFrame,
        features_and_labels,
        **kwargs) -> FeaturesWithLabels:
    features, targets, latent = extract_features(df, features_and_labels, **kwargs)
    labels = extract_labels(df, features_and_labels, **kwargs)
    sample_weights = call_if_not_none(get_pandas_object(df, features_and_labels.sample_weights, **kwargs), 'dropna')
    gross_loss = call_if_not_none(get_pandas_object(df, features_and_labels.gross_loss, **kwargs), 'dropna')

    # do some sanity check for any non numeric values in any of the data frames
    for frame in [features, labels, targets, sample_weights, gross_loss]:
        if frame is not None:
            # we could have nested arrays so we need to use the un-nested values
            values = flatten_nested_list(frame._.values, np.max)
            max_value = max([v.max() for v in values])

            if np.isscalar(max_value) and np.isinf(max_value):
                _log.warning(f"features containing infinit number\n"
                             f"{frame[frame.apply(lambda r: np.isinf(r.values).any(), axis=1)]}")
                frame.replace([np.inf, -np.inf], np.nan, inplace=True)
                frame.dropna(inplace=True)

    # now get the common index and return the filtered data frames
    common_index = intersection_of_index(features, labels, targets, sample_weights, gross_loss)

    return FeaturesWithLabels(
        FeaturesWithRequiredSamples(
            tuple([f.loc[common_index] for f in features]) if isinstance(features, tuple) else features.loc[common_index],
            len(df) - len(features) + 1,
            len(features.columns)
        ),
        labels.loc[common_index],
        loc_if_not_none(latent, common_index),
        loc_if_not_none(targets, common_index),
        loc_if_not_none(sample_weights, common_index),
        loc_if_not_none(gross_loss, common_index)
    )


def extract_features(df: pd.DataFrame, features_and_labels, **kwargs) -> FeaturesWithTargets:
    if isinstance(features_and_labels.features, tuple):
        # allow multiple feature sets i.e. for multi input layered networks
        features = MultiFrameDecorator([get_pandas_object(df, f, **kwargs).dropna() for f in features_and_labels.features], True)
    else:
        features = get_pandas_object(df, features_and_labels.features, **kwargs).dropna()

    targets = call_if_not_none(get_pandas_object(df, features_and_labels.targets, **kwargs), 'dropna')
    latent = call_if_not_none(get_pandas_object(df, features_and_labels.latent, **kwargs), 'dropna')
    common_index = intersection_of_index(features, targets)

    if len(features) <= 0:
        raise ValueError("not enough data!")

    return FeaturesWithTargets(
        features.loc[common_index],
        loc_if_not_none(targets, common_index),
        loc_if_not_none(latent, common_index)
    )


def extract_labels(df: pd.DataFrame, features_and_labels, **kwargs) -> pd.DataFrame:
    labels = get_pandas_object(df, features_and_labels.labels, **kwargs).dropna()

    if features_and_labels.label_type is not None:
        labels = labels.astype(features_and_labels.label_type)

    return labels
