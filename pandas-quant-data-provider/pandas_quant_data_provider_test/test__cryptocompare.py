from unittest import TestCase

import pandas as pd

import pandas_quant_data_provider as dp


class TestCryptoCompare(TestCase):

    def test_price_history(self):
        df = dp.fetch(dp.CryptoCompareSymbol("day", "SHIB", "USD"))
        self.assertGreater(df.shape[0], 500)

