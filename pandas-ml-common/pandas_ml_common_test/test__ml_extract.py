from unittest import TestCase

import numpy as np
import pandas as pd

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

    def test__features_and_labels_extraction_dropna(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = TEST_DF.pct_change()

        """when extracting features and labels"""
        flw = df.ML.extract(FeaturesLabels(features="Open", labels="Close")).extract_features_labels_weights()

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(["Open"], flw.features[0].columns.tolist())
        self.assertEqual(["Close"], flw.labels[0].columns.tolist())
        self.assertEqual(len(df) - 1, len(flw.features[0]))
        self.assertEqual(len(df) - 1, len(flw.labels[0]))

    def test__features_tail_extraction(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = TEST_DF.copy()

        """when extracting features and labels"""
        flw = df.ML.extract(FeaturesLabels(features="Open", labels="Close")).extract_features(tail=2)

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(["Open"], flw.features[0].columns.tolist())
        self.assertEqual(2, len(flw.features[0]))

    def test__features_postprocessor_extraction(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = pd.DataFrame({"a": np.ones(10)})

        """when extracting features and labels"""
        flw = df.ML.extract(
            FeaturesLabels(
                features=[["a"], ["a"]],
                features_postprocessor=lambda df: df + 1,
                labels="a"
            )
        ).extract_features()

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(2, flw.features[0].max().item())
        self.assertEqual(2, flw.features[1].max().item())

    def test__features_multiple_postprocessor_extraction(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = pd.DataFrame({"a": np.ones(10)})

        """when extracting features and labels"""
        flw = df.ML.extract(
            FeaturesLabels(
                features=[["a"], ["a"]],
                features_postprocessor=[lambda df: df + 1, lambda df: df + 2],
                labels="a"
            )
        ).extract_features()

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(2, flw.features[0].max().item())
        self.assertEqual(3, flw.features[1].max().item())
