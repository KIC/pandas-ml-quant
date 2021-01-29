from unittest import TestCase
from pandas_quant_data_provider import quant_data as qd
import pandas as pd


class TestInvestingPlugin(TestCase):

    def test_ambiguous_country(self):
        with self.assertLogs(level='WARN') as cm:
            df0 = qd.load("AAPL", force_provider='investing')
            self.assertIn("default to united states", cm.output[-1])

    def test_dedicated_country(self):
        df0 = qd.load("AAPL/united states", force_provider='investing')
        df1 = qd.load("AAPL/argentina", force_provider='investing')

        self.assertLessEqual(df0.index[0], pd.datetime(1980, 12, 12))
        self.assertLessEqual(df1.index[0], pd.datetime(2010, 7, 30))
