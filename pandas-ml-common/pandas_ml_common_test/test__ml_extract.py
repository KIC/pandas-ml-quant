from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_common import MlTypes, FeaturesLabels
from pandas_ml_common.utils.column_lagging_utils import lag_columns
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

    def test__multiple_features_postprocessor_extraction(self):
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

    def test__multiple_features_lazy_postprocessor_extraction(self):
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

    def test__multiple_features_single_postprocessor_extraction(self):
        """given a patched dataframe"""
        df: MlTypes.PatchedDataFrame = pd.DataFrame({"a": np.ones(10)})

        """when extracting features and labels"""
        flw = df.ML.extract(
            FeaturesLabels(
                features=[["a"], ["a"]],
                features_postprocessor=[lambda df: df + 1],
                labels="a"
            )
        ).extract_features()

        """then we have the expeced shapes for features and labels"""
        self.assertEqual(2, flw.features[0].max().item())
        self.assertEqual(1, flw.features[1].max().item())

    def test__min_required_samples(self):
        df = TEST_DF.copy()

        extractor = df.ML.extract(
            FeaturesLabels(
                features=[lambda df: lag_columns(df["Close"].pct_change(), range(10))],
                labels=[lambda df: df["Close"].pct_change().shift(-1)]
            )
        )

        self.assertEqual(11, extractor.extract_features().min_required_samples)
        self.assertEqual(11, extractor.extract_features_labels_weights().features_with_required_samples.min_required_samples)

    def test__min_required_samples2(self):
        idx = pd.date_range("2020-01-01", "2020-01-31", freq='H')
        df = pd.DataFrame(np.ones((len(idx), 2)), index=idx)

        extractor = df.ML.extract(
            FeaturesLabels(
                features=[lambda df: df.pct_change()],
                features_postprocessor=lambda df: lag_columns(df.dropna().resample('D').apply(list), 10)
            )
        )

        self.assertEqual(31 - 10 + 1, len(extractor.extract_features().features[0]))
        self.assertEqual(24 * 9 + 1, extractor.extract_features().min_required_samples)
