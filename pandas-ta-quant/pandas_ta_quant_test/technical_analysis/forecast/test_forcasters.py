from unittest import TestCase

from pandas_ta_quant.technical_analysis import ta_sarimax, ta_sma
from pandas_ta_quant_test.config import DF_TEST


class TestForecast(TestCase):

    def test_sarimax(self):
        df = DF_TEST[-65:]

        df = ta_sarimax(ta_sma(df["Close"], 2).pct_change().dropna())
        print(df)
        self.assertAlmostEqual(0.0014875, df.iloc[-1, -1], 6)


