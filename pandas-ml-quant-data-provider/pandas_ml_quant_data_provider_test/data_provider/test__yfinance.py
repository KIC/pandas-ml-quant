from unittest import TestCase

import pandas as pd

import pandas_ml_quant_data_provider as dp


class TestYFinance(TestCase):

    def test_simple_download(self):
        df = dp.fetch("XXII")

        self.assertGreater(len(df), 1)
        self.assertNotIsInstance(df.columns, pd.MultiIndex)
        self.assertNotIsInstance(df.index, pd.MultiIndex)

    def test_multi_download(self):
        df1 = dp.fetch(["AAPl", "MSFT"])
        df2 = dp.fetch([["AAPl", "MSFT"]])

        self.assertGreater(len(df1), 1)
        self.assertGreater(len(df2), 1)
        self.assertIsInstance(df1.columns, pd.MultiIndex)
        self.assertNotIsInstance(df1.index, pd.MultiIndex)
        self.assertIsInstance(df2.index, pd.MultiIndex)
        self.assertNotIsInstance(df2.columns, pd.MultiIndex)

