from unittest import TestCase

from pandas_ml_quant.model.summary import WeightedClassificationSummary
from pandas_ml_quant_test.config import DF_TEST_MULTI_CASS


class TestSummary(TestCase):

    def test_multi_class_summary(self):
        df = DF_TEST_MULTI_CASS

        s = WeightedClassificationSummary(df)

        print(s.confusion_indices)
        print(s.df_gross_loss.tail())