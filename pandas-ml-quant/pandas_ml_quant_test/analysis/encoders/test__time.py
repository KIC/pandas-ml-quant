from unittest import TestCase

from pandas_ml_quant.analysis import ta_decimal_year
from pandas_ml_quant_test.config import DF_TEST


class TestTimeEncoders(TestCase):

    def test_time_encoder(self):
        df = DF_TEST["Close"].copy()
        dec = ta_decimal_year(df)

        print(df.index[0], df.index[-1])
        print(dec[0], dec[-1])

        self.assertAlmostEqual(1993.0765027322404, dec[0])
        self.assertAlmostEqual(2019.9234972677596, dec[-1])

