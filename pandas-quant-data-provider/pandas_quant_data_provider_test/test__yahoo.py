from unittest import TestCase

from pandas_quant_data_provider.data_provider.yf import YahooSymbol
from pandas_quant_data_provider.utils.options import calc_greeks
import pandas_quant_data_provider as pd


class TestYahoo(TestCase):

    def test_default_yahoo(self):
        import pandas_quant_data_provider as dp
        df = dp.fetch("SPY")
        self.assertGreater(df.shape[0], 1000)

    def test_price_history(self):
        df = YahooSymbol("SPY").fetch_price_history()
        self.assertGreater(df.shape[0], 1000)

    def test_option_chain(self):
        chains = pd.fetch_option_chain(YahooSymbol("SPY"), 5)
        self.assertListEqual(
            ['call_contract', 'call_bid', 'call_ask', 'call_last', 'call_IV', 'strike', 'dist_pct_spot', 'put_contract', 'put_bid', 'put_ask', 'put_last', 'put_IV'],
            chains.columns.to_list()
        )
        print(chains)

        # test greeks
        greeks = calc_greeks(chains, ['put_bid'], ['call_bid'])
