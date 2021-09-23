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
        self.assertEqual(flw.features[0].columns.tolist(), ["Open"])
        self.assertEqual(flw.labels[0].columns.tolist(), ["Close"])
