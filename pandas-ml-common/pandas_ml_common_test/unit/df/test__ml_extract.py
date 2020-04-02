from unittest import TestCase

from pandas_ml_common_test.config import TEST_DF


class TestMLExtraction(TestCase):

    def test__mock_features_and_labels(self):
        df = TEST_DF.copy()

        self.assertEqual(len(df._.extract(lambda df: df["Close"])), len(df))

