from unittest import TestCase

from pandas_ml_quant_test.config import DF_TEST
from pandas_ml_utils import PostProcessedFeaturesAndLabels, Constant, np
from pandas_ml_utils.ml.data.extraction.features_and_labels_extractor import FeaturesWithLabels


class TestFeaturePostProcesor(TestCase):

    def test_feature_post_processing(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df._.extract(
            PostProcessedFeaturesAndLabels(
                features=[
                    "Close",
                    lambda df: df["Close"].ta.trix(),
                ],
                feature_post_processor=[
                    lambda df: df.ta.rnn(5)
                ],
                labels=[
                    Constant(0)
                ],
                targets=[
                    lambda df: df["Close"]
                ],
            )
        )

        self.assertEqual((6674, 10), fl.features_with_required_samples.features.shape)
        self.assertEqual((6674, 1), fl.labels.shape)
        self.assertEqual((6674, 5, 2), fl.features_with_required_samples.features._.values.shape)

    def test_feature_and_label_post_processing(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df._.extract(
            PostProcessedFeaturesAndLabels(
                features=[
                    "Close",
                    lambda df: df["Close"].ta.trix(),
                ],
                feature_post_processor=[
                    lambda df: df.ta.rnn(5),
                    lambda df: df.ta.rnn(3),
                ],
                labels=[
                    Constant(0)
                ],
                labels_post_processor=[
                    lambda df: df.ta.rnn(4),
                ],
                targets=[
                    lambda df: df["Close"]
                ],
            )
        )

        self.assertEqual((6672, 30), fl.features_with_required_samples.features.shape)
        self.assertEqual((6672, 4), fl.labels.shape)
        # FIXME implement chained lagging: self.assertEqual((6674, 3, 5, 2), fl.features_with_required_samples.features._.values.shape)

    def test_post_row_standardisation(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df._.extract(
            PostProcessedFeaturesAndLabels(
                features=[
                    lambda df: df["Close"].ta.log_returns(),
                    lambda df: df["Close"].ta.trix(),
                    lambda df: df["Close"].ta.rsi(),
                ],
                feature_post_processor=[
                    lambda df: df.ta.rnn(20),
                    lambda df: df.ta.normalize_row('minmax01', level=1)
                ],
                labels=[Constant(0)],
            )
        )

        f = fl.features_with_required_samples.features

        self.assertAlmostEqual(1, f.max(axis=1).values.max())
        self.assertAlmostEqual(0, f.min(axis=1).values.max())
        self.assertEqual(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
             56, 57, 58, 59},
            set(f.apply(np.argmax, axis=1).values)
        )