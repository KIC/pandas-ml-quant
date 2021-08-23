from unittest import TestCase

from pandas_quant_data_provider.data_provider.yf import YahooSymbol


class TestYahoo(TestCase):

    def test_default_yahoo(self):
        import pandas_quant_data_provider as dp
        df = dp.fetch("SPY")
        self.assertGreater(df.shape[0], 1000)

    def test_price_history(self):
        df = YahooSymbol("SPY").fetch_price_history()
        self.assertGreater(df.shape[0], 1000)

    def test_option_chain(self):
        chains = YahooSymbol("SPY").fetch_option_chain( 5)
        print(chains.columns.tolist())
        print(chains)