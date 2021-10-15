import numpy as np
import pandas as pd
from unittest import TestCase

from pandas_ml_common.preprocessing.features_labels import Extractor, FeaturesLabels

DF = pd.DataFrame({
    (1, 1): np.ones(10),
    (1, 2): np.ones(10),
    "(2, 1)": np.ones(10),
    "(2, 2)": np.ones(10),
})


class TestFeaturesLabels(TestCase):

    def test__extract_features_and_labels_single(self):
        extractor = Extractor(
            DF,
            FeaturesLabels(
                features=(1, 2),
                labels="(2, 2)"
            )
        )

        features = extractor.extract_features().features
        labels = extractor.extract_labels().labels

        print(features)
        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (10, 1))

    def test__extract_features_and_labels_list(self):
        extractor = Extractor(
            DF,
            FeaturesLabels(
                features=[(1, 1), "(2, 1)"],
                labels=[(1, 2), "(2, 2)"],
                labels_postprocessor=[None],
                label_type=int,
                reconstruction_targets=[(1, 1)],
            )
        )

        features = extractor.extract_features().features
        labels = extractor.extract_labels().labels
        print(labels)

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].shape, (10, 2))

        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].shape, (10, 2))
        self.assertEqual(labels[0].max().max(), 1)

    def test__extract_features_and_labels_multiple_lists(self):
        extractor = Extractor(
            DF,
            FeaturesLabels(
                features=[[(1, 1), "(2, 1)"], (1, 1)],
                features_postprocessor=[lambda df: df + 1, lambda df: df + 2],
                labels=[[(1, 2), "(2, 2)"], [(1, 2)]],
                labels_postprocessor=lambda df: df + 3,
                label_type=[int, float]
            )
        )

        features = extractor.extract_features().features
        labels = extractor.extract_labels().labels

        self.assertEqual(len(features), 2)
        self.assertEqual(features[0].shape, (10, 2))
        self.assertEqual(features[1].shape, (10, 1))

        self.assertEqual(features[0].max().max(), 2)
        self.assertEqual(features[1].max().max(), 3)

        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0].shape, (10, 2))
        self.assertEqual(labels[1].shape, (10, 1))

        self.assertEqual(labels[0].max().max(), 4)
        self.assertEqual(labels[1].max().max(), 4)

    def test__extract_inf(self):
        with self.assertLogs(level='WARN') as cm:
            flw = Extractor(
                pd.DataFrame({"a": [1, 2, 4, np.nan, np.inf]}),
                FeaturesLabels(
                    features=["a"],
                    labels=["a"]
                )
            ).extract_features_labels_weights()

            self.assertIn("frames containing infinit numbers", cm.output[0])

        self.assertEqual(3, len(flw.features[0]))

    def test__extract_features_and_labels_loc(self):
        extractor = Extractor(
            DF,
            FeaturesLabels(
                features=[[(1, 1), "(2, 1)"], (1, 1)],
                labels=[[(1, 2), "(2, 2)"], [(1, 2)]],
            )
        )

        frames = extractor.extract_features_labels_weights().loc[[0,1,3]]
        self.assertEqual(3, len(frames.features[0]))
        self.assertEqual(3, len(frames.features[1]))
        self.assertEqual(3, len(frames.labels[0]))
        self.assertEqual(3, len(frames.labels[1]))