from unittest import TestCase

from pandas_ta_quant_test.config import DF_TEST
from pandas_ta_quant.portfolio.price import PriceTimeSeries


class TestPriceTimeSeries(TestCase):

    def test_adding_prices(self):
        ts = PriceTimeSeries()

        ts.push_bar('MSFT', '2020-01-01', 12, 12.3, 11.8, 11.9, 0)
        ts.push_bar('MSFT', '2020-01-03', 13, 13.3, 12.8, 12.9, 0)
        ts.push_bar('MSFT', '2020-01-06', 14, 14.3, 13.8, 13.9, 0)

        self.assertEqual(ts.get_price('MSFT', '2020-01-03')[1][0], 11.9)
        self.assertEqual(ts.get_price('MSFT', '2020-01-03 00:00:00.0001')[1][0], 13)
        self.assertEqual(ts.get_price('MSFT', '2020-01-03 01:00:00')[1][0], 13)
        self.assertEqual(ts.get_price('MSFT', '2020-01-05')[1][0], 12.9)

    def test_reading_dataframe(self):
        ts = PriceTimeSeries.from_dataframe(DF_TEST, 'SPY', ohlc=["Open", "High", "Low", "Close"])

        # 1994-07-08,44.640625,44.984375,44.625000,44.906250,28.198448,148400
        # 1994-07-11,44.937500,45.015625,44.531250,44.750000,28.100328,124000
        time, (bid, ask) = ts.get_price('SPY', '1994-07-09')
        print(time, bid, ask)

        self.assertAlmostEqual(bid, 44.906250)
        self.assertAlmostEqual(ts.get_price('SPY', '1994-07-11')[1][0], 44.906250)
        self.assertAlmostEqual(ts.get_price('SPY', '1994-07-12')[1][0], 44.750000)
