from unittest import TestCase

import pandas as pd

from pandas_quant_data_provider import quant_data as qd


class TestQuantData(TestCase):

    def test_yahoo(self):
        df0 = qd.load("AAPL", force_provider='yahoo')
        df1 = qd.load(["AAPL", "MSFT"], force_provider='yahoo')
        df2 = qd.load(["AAPL", "MSFT"], ["AL", "DAT"], force_provider='yahoo')
        df3 = qd.load([["AAPL", "MSFT"]], force_provider='yahoo')

        self.assertNotIsInstance(df0.columns, pd.MultiIndex)
        self.assertNotIsInstance(df0.index, pd.MultiIndex)

        self.assertIsInstance(df1.columns, pd.MultiIndex)
        self.assertNotIsInstance(df1.index, pd.MultiIndex)

        self.assertIsInstance(df2.columns, pd.MultiIndex)
        self.assertIsInstance(df2.index, pd.MultiIndex)

        self.assertNotIsInstance(df3.columns, pd.MultiIndex)
        self.assertIsInstance(df3.index, pd.MultiIndex)
