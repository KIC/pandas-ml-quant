from unittest import TestCase

from pandas_ml_common.preprocessing.features_labels import FeaturesWithLabels
from pandas_ml_quant_test.config import DF_TEST
from pandas_ml_utils import FeaturesLabels, Constant, np


class TestFeaturePostProcesor(TestCase):

    def test_feature_post_processing(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesLabels(
                features=[
                    "Close",
                    lambda df: df["Close"].ta.trix(),
                ],
                features_postprocessor=[
                    lambda df: df.ta.rnn(5)
                ],
                labels=[
                    Constant(0)
                ],
                reconstruction_targets=[
                    lambda df: df["Close"]
                ],
            )
        ).extract_features_labels_weights()

        self.assertEqual(1, len(fl.features))
        self.assertEqual((6674, 10), fl.features_with_required_samples.features[0].shape)
        self.assertEqual((6674, 5, 2), fl.features_with_required_samples.features[0].ML.values.shape)
        self.assertEqual((6763, 1), fl.labels[0].shape)

    def test_feature_and_label_post_processing(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesLabels(
                features=[
                    "Close",
                    lambda df: df["Close"].ta.trix(),
                ],
                features_postprocessor=[
                    lambda df: df.ta.rnn(5).ta.rnn(3)
                ],
                labels=[
                    Constant(0)
                ],
                labels_postprocessor=[
                    lambda df: df.ta.rnn(4),
                ],
                reconstruction_targets=[
                    lambda df: df["Close"]
                ],
            )
        ).extract_features_labels_weights()

        self.assertEqual(1, len(fl.features))
        self.assertEqual((6672, 30), fl.features[0].shape)
        self.assertEqual((6760, 4), fl.labels[0].shape)
        # FIXME implement chained lagging: self.assertEqual((6674, 3, 5, 2), fl.features_with_required_samples.features.ML.values.shape)

    def test_empty_post_prcessor(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesLabels(
                features=[
                    lambda df: df["Close"].ta.log_returns(),
                ],
                labels=[Constant(0)],
            )
        ).extract_features_labels_weights()

        f = fl.features_with_required_samples.features

        self.assertEqual(1, len(f))
        self.assertEqual((6762, 1), f[0].shape)
        np.testing.assert_array_almost_equal(df[["Close"]].ta.log_returns().dropna().values, f[0].values)

    def test_post_row_standardisation(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesLabels(
                features=[
                    lambda df: df["Close"].ta.log_returns(),
                    lambda df: df["Close"].ta.trix(),
                    lambda df: df["Close"].ta.rsi(),
                ],
                features_postprocessor=[
                    lambda df: df.ta.rnn(20).ta.normalize_row('minmax01', level=1)
                ],
                labels=[Constant(0)],
            )
        ).extract_features_labels_weights()

        f = fl.features
        self.assertEqual(1, len(f))
        self.assertEqual((6659, 20 * 3), f[0].shape)
        self.assertAlmostEqual(1, f[0].max(axis=1).values.max())
        self.assertAlmostEqual(0, f[0].min(axis=1).values.max())
        self.assertEqual(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
             56, 57, 58, 59},
            set(f[0].apply(np.argmax, axis=1).values)
        )

    def test_multiple_features_post_processing(self):
        df = DF_TEST.copy()

        fl: FeaturesWithLabels = df.ML.extract(
            FeaturesLabels(
                features=[
                    [lambda df: df["Close"].ta.log_returns(), lambda df: df["Close"].ta.trix(), lambda df: df["Close"].ta.rsi()],
                    [lambda df: df["Close"].ta.rsi()],
                ],
                features_postprocessor=[
                    lambda df: df.ta.rnn(20).ta.normalize_row('minmax01', level=1),
                    lambda df: df.ta.rnn(10)
                ],
                labels=[Constant(0)],
            )
        ).extract_features_labels_weights()

        f = fl.features_with_required_samples.features
        self.assertEqual(2, len(f))

        a, b = f

        # test a
        self.assertEqual((6659, 20 * 3), a.shape)
        self.assertAlmostEqual(1, a.max(axis=1).values.max())
        self.assertAlmostEqual(0, a.min(axis=1).values.max())
        self.assertEqual(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
             29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
             56, 57, 58, 59},
            set(a.apply(np.argmax, axis=1).values)
        )

        # test b
        self.assertEqual((6659, 10), b.shape)
        np.testing.assert_array_almost_equal(
            b.values[-1],
            [0.580079, 0.561797, 0.501477, 0.58181 , 0.716154, 0.789371, 0.762768, 0.74797 , 0.687273, 0.666173]
        )

    def test_multi_fearutes_with_empy_post_processor(self):
        df = DF_TEST.copy()

        with df.model() as m:
            fnl = m.extract(
                FeaturesLabels(
                    features=[
                        [
                            lambda df: df.ta.dist_opex(),
                            lambda df: df.ta.rsi(),
                        ],
                        [
                            lambda df: df.ta.hf_lf_vola(periods=range(3, 5))
                        ]
                    ],
                    features_postprocessor=[
                            lambda df: df.ta.rnn(2), None
                    ],
                    labels=[
                        lambda df: df["Close"].ta.log_returns().shift(-1).rename("future_log_return")
                    ]
                )
            ).extract_features_labels_weights()

        f = fnl.features
        self.assertEqual(2, len(f))
        self.assertEqual((6750, 14), fnl.features[0].shape)
        self.assertEqual((6750, 2), fnl.features[1].shape)
