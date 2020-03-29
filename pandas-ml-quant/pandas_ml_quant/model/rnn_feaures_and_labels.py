from pandas_ml_quant.analysis.indicators import ta_rnn
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils.ml.data.extraction import extract_with_post_processor


class RNNFeaturesAndLabels(FeaturesAndLabels):

    def __init__(self, features_and_labels: FeaturesAndLabels, lags):
        super().__init__(
            features=extract_with_post_processor(
                features_and_labels.features,
                lambda df: ta_rnn(df, lags)
            ),
            labels=features_and_labels.labels,
            targets=features_and_labels.targets,
            sample_weights = features_and_labels.sample_weights,
            gross_loss = features_and_labels.gross_loss,
            label_type = features_and_labels.label_type,
            **features_and_labels.kwargs
        )
