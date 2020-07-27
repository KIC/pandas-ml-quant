from typing import Tuple, List, Callable

import pandas as pd
import numpy as np
import logging

from pandas_ml_common import get_pandas_object, Typing
from pandas_ml_common.utils import intersection_of_index, loc_if_not_none
from pandas_ml_common.utils.callable_utils import call_if_not_none

_log = logging.getLogger(__name__)


def extract_feature_labels_weights(
        df: Typing.PatchedDataFrame,
        features_and_labels,
        **kwargs) -> Tuple[Tuple[pd.DataFrame, int], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = get_pandas_object(df, features_and_labels.features, **kwargs).dropna()
    labels = get_pandas_object(df, features_and_labels.labels, **kwargs).dropna()
    targets = call_if_not_none(get_pandas_object(df, features_and_labels.targets, **kwargs), 'dropna')
    sample_weights = call_if_not_none(get_pandas_object(df, features_and_labels.sample_weights, **kwargs), 'dropna')
    gross_loss = call_if_not_none(get_pandas_object(df, features_and_labels.gross_loss, **kwargs), 'dropna')

    if features_and_labels.label_type is not None:
        labels = labels.astype(features_and_labels.label_type)

    for frame in [features, labels, targets, sample_weights, gross_loss]:
        if frame is not None:
            max = frame._.values.max()

            if np.isscalar(max) and np.isinf(max):
                _log.warning("features containing infinit number\n", frame[frame.apply(lambda r: np.isinf(r.values).any(), axis=1)])
                frame.replace([np.inf, -np.inf], np.nan, inplace=True)
                frame.dropna(inplace=True)

    common_index = intersection_of_index(features, labels, targets, sample_weights, gross_loss)

    return (
        (features.loc[common_index], len(df) - len(features) + 1),
        labels.loc[common_index],
        loc_if_not_none(targets, common_index),
        loc_if_not_none(sample_weights, common_index),
        loc_if_not_none(gross_loss, common_index)
    )


def extract_features(df: pd.DataFrame, features_and_labels, **kwargs) -> Tuple[List, pd.DataFrame, pd.DataFrame]:
    features = get_pandas_object(df, features_and_labels.features, **kwargs).dropna()
    targets = call_if_not_none(get_pandas_object(df, features_and_labels.targets, **kwargs), 'dropna')
    common_index = intersection_of_index(features, targets)

    if len(features) <= 0:
        raise ValueError("not enough data!")

    return (
        features_and_labels.label_columns,
        features.loc[common_index],
        loc_if_not_none(targets, common_index)
    )


def extract(features_and_labels, df, extractor, *args, **kwargs):
    return features_and_labels(df, extractor, *args, **kwargs)


def extract_with_post_processor(list, postprocessor: Callable):
    def extractor(df, **kwargs):
        extraction = get_pandas_object(df, list, **kwargs)
        return get_pandas_object(extraction, postprocessor, **kwargs)

    return extractor

