from unittest import TestCase

from pandas_ml_quant.analysis import ta_trend_lines
from pandas_ml_quant_test.config import DF_TEST


class TestSupportResistence(TestCase):

    def test_trend_lines(self):
        df = DF_TEST["Close"][-200:].copy()

        tl = ta_trend_lines(df)
        print(tl)