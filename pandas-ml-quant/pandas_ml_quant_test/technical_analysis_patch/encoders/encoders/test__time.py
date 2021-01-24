from unittest import TestCase

from pandas_ta_quant.technical_analysis import ta_decimal_year, ta_sinusoidal_week_day
from pandas_ta_quant_test.config import DF_TEST, DF_TEST_MULTI_ROW
import pandas as pd


class TestTimeEncoders(TestCase):

    def test_time_encoder(self):
        df = DF_TEST["Close"].copy()
        dec = ta_decimal_year(df)

        print(df.index[0], df.index[-1])
        print(dec[0], dec[-1])

        self.assertAlmostEqual(1993.0765027322404, dec[0])
        self.assertAlmostEqual(2019.9234972677596, dec[-1])

    def test_time_encoder_multi_index_row(self):
        df = DF_TEST_MULTI_ROW.copy()
        dec = ta_decimal_year(df)

        print(df.index[0], df.index[-1])
        print(dec[0], dec[-1])

        self.assertIsInstance(dec.index, pd.MultiIndex)
        self.assertAlmostEqual(1993.0765027322404, dec[0])
        self.assertAlmostEqual(2020.2349726775956, dec[-1])

    def test_sinusidal_week(self):
        df = DF_TEST_MULTI_ROW.copy()
        dec = ta_sinusoidal_week_day(df)

        print(df.index[0], df.index[-1])
        print(dec[0], dec[-1])

        self.assertAlmostEqual(-0.8660254037844384, dec[0])
        self.assertAlmostEqual(-0.8660254037844384, dec[-1])
        self.assertAlmostEqual(0, dec[-5])

