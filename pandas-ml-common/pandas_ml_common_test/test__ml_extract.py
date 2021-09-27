from unittest import TestCase

from pandas_ml_common import MlTypes, FeaturesLabels
from pandas_ml_common_test.config import TEST_DF


class TestMLExtraction(TestCase):

    def test__features_and_labels_extraction(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()

        """when extracting features and labels"""
        flw = df.ML.extract(FeaturesLabels(features="Open", labels="Close")).extract_features_labels_weights()

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(["Open"], flw.features[0].columns.tolist())
        self.assertEqual(["Close"], flw.labels[0].columns.tolist())

    def test__features_tail_extraction(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()

        """when extracting features and labels"""
        flw = df.ML.extract(FeaturesLabels(features="Open", labels="Close")).extract_features(tail=2)

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(["Open"], flw.features[0].columns.tolist())
        self.assertEqual(2, len(flw.features[0]))
