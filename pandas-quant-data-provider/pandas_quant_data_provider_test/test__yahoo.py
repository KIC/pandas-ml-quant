from unittest import TestCase


class TestYahoo(TestCase):

    def test_yahoo(self):
        import pandas_quant_data_provider as dp
        df = dp.fetch("SPY")
        self.assertGreater(df.shape[0], 1000)
