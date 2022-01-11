from unittest import TestCase

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

