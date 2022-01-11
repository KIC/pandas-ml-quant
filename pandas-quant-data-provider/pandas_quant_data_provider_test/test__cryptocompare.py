from unittest import TestCase

import pandas_quant_data_provider as dp
from pandas_quant_data_provider.data_provider.cryptocompare.cryptocompare_provider import CryptoCompareSymbol


class TestCryptoCompare(TestCase):

    def test_price_history(self):
        df = dp.fetch(CryptoCompareSymbol("day", "SHIB", "USD"))
        self.assertGreater(df.shape[0], 500)

