from unittest import TestCase

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
