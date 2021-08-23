from unittest import TestCase

from pandas_ta_quant.technical_analysis import ta_gkyz_volatility, ta_cc_volatility, ta_repeat
from pandas_ta_quant_test.config import DF_TEST


class TestMeta(TestCase):

    def test_repeat(self):
        df = DF_TEST[-100:]

        def test(df, param):
            return (ta_gkyz_volatility(df, period=param) / ta_cc_volatility(df["Close"], period=param) - 1)\
                .rename(f"{param}")

        result = ta_repeat(df, test, range(2, 10), multiindex="HF/RF Vola Ratio").dropna()
        #print(result)

        self.assertEqual(result.shape, (91, 8))
