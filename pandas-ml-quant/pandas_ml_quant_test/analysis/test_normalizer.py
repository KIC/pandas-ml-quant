from unittest import TestCase

from pandas_ml_quant.analysis import ta_ma_ratio
from pandas_ml_quant_test.config import DF_TEST


class TestNormalizer(TestCase):

    def test_ma_ratio(self):
        df = DF_TEST

        ma_ration_scaled = ta_ma_ratio(df)
        print(ma_ration_scaled.columns)
        self.assertTrue(True)