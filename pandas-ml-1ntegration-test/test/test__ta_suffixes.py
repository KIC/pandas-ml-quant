from unittest import TestCase
from pandas_ml_quant_test.config import DF_TEST_MULTI
from pandas_ml_quant import pd


class TestTASuffixes(TestCase):

    def test_multiindex_suffix(self):
        df = DF_TEST_MULTI
        res = df.loc[:, (slice(None), "Close")].ta.rsi()

        print(res)