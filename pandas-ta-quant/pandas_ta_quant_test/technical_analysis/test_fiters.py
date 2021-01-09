from unittest import TestCase

from pandas_ta_quant.technical_analysis import ta_multi_ma
from pandas_ta_quant_test.config import DF_TEST


class TestFilter(TestCase):

    def test_multi_ma(self):
        df = DF_TEST
        mma = ta_multi_ma(df)

        self.assertEqual(len(mma.columns), 30)

