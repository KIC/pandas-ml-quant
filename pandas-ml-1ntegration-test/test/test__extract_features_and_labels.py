from unittest import TestCase

import numpy as np

import pandas_ml_quant
from pandas_ml_common.utils.serialization_utils import serialize, deserialize
from pandas_ml_utils import FeaturesAndLabels, PostProcessedFeaturesAndLabels
from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels
from test.config import DF_TEST

print(pandas_ml_quant.__version__)


class TestExtractionOfFeaturesAndLabels(TestCase):

    def test_extract_in_rnn_shape(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesAndLabels(
                features=[
                    lambda df: df["Close"].ta.rsi(14).ta.rnn(280),
                    lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).ta.rnn(280)
                ],
                labels=[
                    lambda df, forecasting_time_steps, stddevs: df["Close"].ta.future_bband_quantile(14, forecasting_time_steps, stddev=stddevs, include_mean=True)\
                                                                           .ta.one_hot_encode_discrete(),
                ],
                targets=[
                    lambda df, stddevs: df["Close"].ta.bbands(period=14, stddev=stddevs)[["lower", "mean", "upper"]],
                ],
                gross_loss=lambda df: 1
            ),
            # kwargs of call
            forecasting_time_steps=7,
            stddevs=1.5
        )

        print(fl.features_with_required_samples.features, fl.labels, fl.sample_weights, fl.targets, fl.gross_loss)
        print(fl.features_with_required_samples.features.ML.values.shape, fl.labels.ML.values.shape)
        print(len(df))

        # we need RNN shape to be [row, time_step, feature]
        self.assertEqual((6463, 280, 2), fl.features_with_required_samples.features.ML.values.shape)

        # we have 2 labels each one hot encoded to 10 values
        self.assertEqual((6463, 4), fl.labels.ML.values.shape)
        self.assertEqual((6463, 4), fl.labels.ML.values.squeeze().shape)

        self.assertEqual(len(fl.features_with_required_samples.features), len(fl.labels))
        self.assertLess(len(fl.features_with_required_samples.features), len(df))

        self.assertEqual(fl.gross_loss.values.sum(), len(fl.features_with_required_samples.features))

    def test_extract_in_rnn_shape_two_labels(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesAndLabels(
                features=[
                    lambda df: df["Close"].ta.rsi(14).ta.rnn(280),
                    lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).ta.rnn(280)
                ],
                labels=[
                    lambda df, forecasting_time_steps, stddevs: df["Close"].ta.future_bband_quantile(14, forecasting_time_steps, stddev=stddevs, include_mean=False)\
                                                                           .ta.one_hot_encode_discrete(),
                    lambda df, forecasting_time_steps, stddevs: df["Open"].ta.future_bband_quantile(14, forecasting_time_steps, stddev=stddevs, include_mean=False)\
                                                                          .ta.one_hot_encode_discrete(),
                ],
                targets=[
                    lambda df, stddevs: df["Close"].ta.bbands(period=14, stddev=stddevs, include_mean=False)[["lower", "upper"]],
                ]
            ),
            # kwargs of call
            forecasting_time_steps=7,
            stddevs=1.5
        )

        # we need RNN shape to be [row, time_step, feature]
        self.assertEqual(294, fl.features_with_required_samples.min_required_samples)
        self.assertEqual((6463, 280, 2), fl.features_with_required_samples.features.ML.values.shape)

        # we have 2 labels each one hot encoded to 10 values
        self.assertEqual((6463, 2, 3), fl.labels.ML.values.shape)
        self.assertEqual((6463, 2, 3), fl.labels.ML.values.squeeze().shape)

        self.assertEqual(len(fl.features_with_required_samples.features), len(fl.labels))
        self.assertLess(len(fl.features_with_required_samples.features), len(df))

    def test_serialize_deserialize(self):
        df = DF_TEST.copy()
        file = '/tmp/fnl.dill'

        serialize(FeaturesAndLabels(
                features=[
                    lambda df: df["Close"].ta.rsi(14).ta.rnn(280),
                    lambda df: (df["Volume"] / df["Volume"].ta.ema(14) - 1).ta.rnn(280)
                ],
                labels=[
                    lambda df, forecasting_time_steps, stddevs: df["Close"].ta.future_bband_quantile(14, forecasting_time_steps, stddev=stddevs, include_mean=False)\
                                                                           .ta.one_hot_encode_discrete(),
                    lambda df, forecasting_time_steps, stddevs: df["Open"].ta.future_bband_quantile(14, forecasting_time_steps, stddev=stddevs, include_mean=False)\
                                                                          .ta.one_hot_encode_discrete(),
                ],
                targets=[
                    lambda df, stddevs: df["Close"].ta.bbands(period=14, stddev=stddevs, include_mean=False)[["lower", "upper"]],
                ]
            ),
            file
        )

        fl: FeaturesWithLabels  = df.ML.extract(
            deserialize(file, FeaturesAndLabels),
            # kwargs of call
            forecasting_time_steps=7,
            stddevs=1.5
        )

        # we need RNN shape to be [row, time_step, feature]
        self.assertEqual((6463, 280, 2), fl.features_with_required_samples.features.ML.values.shape)

        # we have 2 labels each one hot encoded to 10 values
        self.assertEqual((6463, 2, 3), fl.labels.ML.values.shape)

    def test_feature_post_processing_pipeline(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            PostProcessedFeaturesAndLabels(
                features=[
                    lambda df: df._["Close"].ta.log_returns()
                ],
                feature_post_processor=[
                    lambda df: df.flatten_columns(),
                    lambda df: df.ta.rnn(2),
                    lambda df: df.ta.normalize_row(normalizer='uniform'),
                ],
                labels=[
                    lambda df: df._["Close"].ta.log_returns().shift(-1)
                ]
            )
        )

        self.assertEqual((6760, 2, 1), fl.features_with_required_samples.features.ML.values.shape)
        np.testing.assert_array_almost_equal(fl.features_with_required_samples.features[-1:].values, np.array([[1.5, 1.0]]))
