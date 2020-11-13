from unittest import TestCase

from pandas_ml_quant_test.config import DF_TEST_MULTI
from pandas_ml_utils import FeaturesAndLabels


class TestFeaturePostProcesor(TestCase):

    def test_test_multi_column_features_extraction(self):
        fl = FeaturesAndLabels(
            features=[
                lambda df: df._["Close"].ta.log_returns().droplevel(0, axis=1),
                lambda df: df._["Close"].ta.rsi().droplevel(0, axis=1),
            ],
            labels=[
                lambda df: df[[("spy", "Close"), ("gld", "Close")]].ta.log_returns().ta.sma(5).shift(-5)
            ]
        )

        DF_TEST_MULTI._.extract(fl)

