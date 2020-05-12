from unittest import TestCase

from pandas_ml_quant import PostProcessedFeaturesAndLabels, Constant
from pandas_ml_quant_test.config import DF_TEST


class TestFeaturePostProcesor(TestCase):

    def test_repr(self):
        fnl = PostProcessedFeaturesAndLabels(
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

        print(repr(repr(fnl).replace('"', '\"')))

        self.assertEqual(
            'PostProcessedFeaturesAndLabels(\t[\'Close\', \'                lambda df: df["Close"].ta.trix(),\\n\'], \t[\'                lambda df: df.ta.rnn(5)\\n\'], \t[\'Constant(0)\'], \tNone, \tNone, \tNone, \tNone, \tNone, \t[\'                lambda df: df["Close"]\\n\'], \tNone, \tNone, \t{})',
            repr(fnl)
        )

    def test_feature_post_processing(self):
        df = DF_TEST.copy()

        (f, min), l, t, w, gl = df._.extract(
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

        self.assertEqual((6674, 10), f.shape)
        self.assertEqual((6674, 1), l.shape)
        self.assertEqual((6674, 5, 2), f._.values.shape)

    def test_feature_and_label_post_processing(self):
        df = DF_TEST.copy()

        (f, min), l, t, w, gl = df._.extract(
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

        self.assertEqual((6672, 30), f.shape)
        self.assertEqual((6672, 4), l.shape)
        # FIXME self.assertEqual((6674, 3, 5, 2), f._.values.shape)

    def test_post_z_standardisation(self):
        # FIXME implement me ...
        pass