from unittest import TestCase

from pandas_ml_utils import FeaturesAndLabels
from test.config import DF_TEST
import pandas_ml_quant


print(pandas_ml_quant.__version__)


class TestExtractionOfFeaturesAndLabels(TestCase):

    def test_extract_in_rnn_shape(self):
        # implement something along the lines
        df = DF_TEST.copy()

        features, labels, weights = df.ml.extract(
            FeaturesAndLabels(
                features=[
                    lambda df: df["Close"].q.ta_rsi().q.ta_shape_for_auto_regression(280),
                    lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).q.ta_shape_for_auto_regression(280)
                ],
                labels=[
                    lambda df, forecasting_time_steps, stddevs: df["Close"].q.ta_future_multiband_bucket(forecasting_time_steps, period=14, stddevs=stddevs)\
                                                                           .q.ta_one_hot_encode_discrete(),
                ],
                targets=[
                    lambda df, stddevs: df["Close"].q.ta_multi_bbands(period=14, stddevs=stddevs),
                ],
                min_required_samples=280
            ),
            # kwargs of call
            forecasting_time_steps=7,
            stddevs=[0.5, 1.5, 2.5, 3.5]
        )

        print(features, labels, weights)
        print(features.ml.values.shape, labels.ml.values.shape)
        print(len(df))

        # we need RNN shape to be [row, time_step, feature]
        self.assertEqual((6463, 280, 2), features.ml.values.shape)

        # we have 2 labels each one hot encoded to 10 values
        self.assertEqual((6463, 1, 10), labels.ml.values.shape)
        self.assertEqual((6463, 10), labels.ml.values.squeeze().shape)

        self.assertEqual(len(features), len(labels))
        self.assertLess(len(features), len(df))

    def test_extract_in_rnn_shape_two_labels(self):
        # implement something along the lines
        df = DF_TEST.copy()

        features, labels, weights = df.ml.extract(
            FeaturesAndLabels(
                features=[
                    lambda df: df["Close"].q.ta_rsi().q.ta_shape_for_auto_regression(280),
                    lambda df: (df["Volume"] / df["Volume"].q.ta_ema(14) - 1).q.ta_shape_for_auto_regression(280)
                ],
                labels=[
                    lambda df, forecasting_time_steps, stddevs: df["Close"].q.ta_future_multiband_bucket(forecasting_time_steps, period=14, stddevs=stddevs)\
                                                                           .q.ta_one_hot_encode_discrete(),
                    lambda df, forecasting_time_steps, stddevs: df["Open"].q.ta_future_multiband_bucket(forecasting_time_steps, period=14, stddevs=stddevs)\
                                                                          .q.ta_one_hot_encode_discrete(),
                ],
                targets=[
                    lambda df, stddevs: df["Close"].q.ta_multi_bbands(period=14, stddevs=stddevs),
                ],
                min_required_samples=280
            ),
            # kwargs of call
            forecasting_time_steps=7,
            stddevs=[0.5, 1.5, 2.5, 3.5]
        )

        # we need RNN shape to be [row, time_step, feature]
        self.assertEqual((6463, 280, 2), features.ml.values.shape)

        # we have 2 labels each one hot encoded to 10 values
        self.assertEqual((6463, 2, 10), labels.ml.values.shape)
        self.assertEqual((6463, 2, 10), labels.ml.values.squeeze().shape)

        self.assertEqual(len(features), len(labels))
        self.assertLess(len(features), len(df))



"""
    FeaturesAndLabels(
    features=["rsi", "volume"],
    labels=["future_bb"],
    sample_weights="loss_weight",
    # targets=lambda df, _, stddevs: df["Close"].ta_multi_bbands(period=14, stddevs=stddevs), # TODO rethink targets
    gross_loss=lambda df: df["gross_loss"],
    # feature_lags=range(20), # FIXME need to be part of the indicators 
    pre_processor=lambda df, forecasting_time_steps, stddevs: pmu.LazyDataFrame( # TODO rethink if preprocessor is the right thing to do
        df,
        # features
        rst             = lambda df: df["Close"].q.ta_rsi(),
        volume          = lambda df: df["Volume"] / df["Volume"].ta_ema(14) - 1,
        
        # lables 
        future_bb       = lambda df: df["Close"].ta_future_multiband_bucket(forecasting_time_steps, period=14, stddevs=stddevs),
        
        # gross loss
        gross_loss      = lambda df: df["Close"].ta_future_multiband_loss(forecasting_time_steps, period=14, stddevs=stddevs),

        # sample loss (weight)
        loss_weight     = lambda df: df["gross_loss"].abs() + 1
    ).to_dataframe()
)

"""
