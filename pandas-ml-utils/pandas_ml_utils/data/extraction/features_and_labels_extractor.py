from typing import Tuple

import pandas as pd
from pandas_ml_common.utils import get_pandas_object


def extract_feature_labels_weights(
        df: pd.DataFrame,
        features_and_labels,
        **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    features = get_pandas_object(df, features_and_labels.features, **kwargs)
    labels = get_pandas_object(df, features_and_labels.labels, **kwargs)
    sample_weights = get_pandas_object(df, features_and_labels.sample_weights, **kwargs)

    return (features, labels, sample_weights)


def extract_features(df: pd.DataFrame, features_and_labels, **kwargs):
    return get_pandas_object(df, features_and_labels.features, **kwargs)

