from unittest import TestCase

from pandas_ml_utils import PostProcessedFeaturesAndLabels, Constant
from pandas_ml_quant_test.config import DF_TEST
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
        # FIXME self.assertEqual((6674, 3, 5, 2), f._.values.shape)

    def test_post_z_standardisation(self):
        # FIXME implement me ...
        pass