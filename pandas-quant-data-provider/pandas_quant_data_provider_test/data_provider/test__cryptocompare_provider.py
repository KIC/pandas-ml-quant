from unittest import TestCase

import pandas as pd

from pandas_quant_data_provider.data_provider.cryptocompare.cryptocompare_provider import CryptoCompareSymbol


class TestCryptoCompareProvider(TestCase):

    def test_fetch_prices(self):
        prices = CryptoCompareSymbol("hour", "SHIB", "USD").fetch_price_history()
        increments = prices.index.to_series().diff().dropna()

        self.assertFalse(prices.index.has_duplicates)
        self.assertEqual(increments.min(), increments.max())

    def test_fetch_price_aggregates(self):
        prices = CryptoCompareSymbol((5, "hour"), "SHIB", "USD").fetch_price_history()
        increments = prices.index.to_series().diff().dropna()

        self.assertFalse(prices.index.has_duplicates)
        self.assertEqual(increments.min(), increments.max())

    def test_str(self):
        self.assertEqual('SHIBUSD', f'{CryptoCompareSymbol("hour", "SHIB", "USD")}')

    def test_max_hist(self):
        dfd = CryptoCompareSymbol("day", "BTC", "USD").fetch_price_history()
        dfh = CryptoCompareSymbol("hour", "BTC", "USD").fetch_price_history()

        # expected 2005-06-23 00:00:00+00:00  vs   2010-09-21 16:00:00+00:00
        # print(dfd.index[0], dfh.index[0])
        self.assertLess(dfd.index[0], pd.to_datetime("2010-09-22", utc=True))
        self.assertLess(dfh.index[0], pd.to_datetime("2010-09-22", utc=True))
