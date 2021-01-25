from unittest import TestCase

from pandas_quant_data_provider import quant_data as qd


class TestQuantData(TestCase):

    def test_provider_selection(self):

        with self.assertLogs(level='WARNING') as cm:
            df = qd.load("AAPL").tail()
            msg = [log for log in cm.output if "Ambiguous symbol has multiple providers" in log]

            self.assertGreaterEqual(len(msg), 1)

        print(df)

