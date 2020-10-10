from unittest import TestCase
from pandas_ml_quant_data_provider import quant_data as qd


class TestInvestingPlugin(TestCase):

    def test_ambiguous_country(self):
        with self.assertLogs(level='WARN') as cm:
            df0 = qd.load("AAPL", force_provider='investing')
            self.assertIn("default to united states", cm.output[-1])


