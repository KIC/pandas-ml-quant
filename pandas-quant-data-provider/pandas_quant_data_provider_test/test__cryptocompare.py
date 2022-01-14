import unittest
from unittest import TestCase

import pandas_quant_data_provider as dp


class TestCryptoCompare(TestCase):

    def test_price_history(self):
        df = dp.fetch(dp.CryptoCompareSymbol("day", "SHIB", "USD"))
        self.assertGreater(df.shape[0], 500)

    @unittest.skip("only test locally")
    def test_joined_symbols(self):
        df = dp.fetch(
            dp.CryptoCompareSymbol("hour", quote="USD").for_coins(
                'ADA', 'BCH', 'BTC', 'DASH', 'EOS', 'ETC', 'ETH', 'GNO', 'LTC', 'QTUM', 'REP', 'USDT', 'XLM', 'XMR', 'XRP',
            ),
        )

        # print(df.shape)
        self.assertEqual(df.shape[1], 135)
        self.assertGreaterEqual(df.shape[0], 39165)
