from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_quant_data_provider import _QuantData as QuantData


class TestQuantData(TestCase):

    class MokedDataProvider(object):

        def __init__(self, value = None):
            self.value = value or 1

        def has_symbol(self, symbol: str):
            return True

        def load(self, symbol: str):
            return pd.DataFrame({"Close": np.ones(10) * self.value}, index=pd.date_range('2020-01-01', '2020-01-10'))

    def test_shape(self):
        qd = QuantData(plugins={"mock": TestQuantData.MokedDataProvider()})

        np.testing.assert_array_equal(qd._init_shape("AAPL"), np.array([["AAPL"]]))
        np.testing.assert_array_equal(qd._init_shape(["AAPL"]), np.array([["AAPL"]]))

        np.testing.assert_array_equal(qd._init_shape("AAPL", "MSFT"), np.array([["AAPL", "MSFT"]]))
        np.testing.assert_array_equal(qd._init_shape(["AAPL", "MSFT"]), np.array([["AAPL", "MSFT"]]))

        np.testing.assert_array_equal(qd._init_shape([["AAPL", "MSFT"]]), np.array([["AAPL"], ["MSFT"]]))
        np.testing.assert_array_equal(qd._init_shape(["AAPL"], ["MSFT"]), np.array([["AAPL"], ["MSFT"]]))
        np.testing.assert_array_equal(qd._init_shape([["AAPL"], ["MSFT"]]), np.array([["AAPL"], ["MSFT"]]))

        np.testing.assert_array_equal(qd._init_shape([["AAPL", "MSFT"], ["TSLA", "NIO"]]),
                                      np.array([["AAPL", "MSFT"], ["TSLA", "NIO"]]))

    def test_indices(self):
        qd = QuantData(plugins={"mock": TestQuantData.MokedDataProvider()})

        frame12 = qd.load("AAPL", "MSFT")
        self.assertIsInstance(frame12.columns, pd.MultiIndex)
        self.assertNotIsInstance(frame12.index, pd.MultiIndex)
        self.assertListEqual(frame12.columns.to_list(), [("AAPL", "Close"), ("MSFT", "Close")])

        frame21 = qd.load([["AAPL", "MSFT"]])
        self.assertIsInstance(frame21.index, pd.MultiIndex)
        self.assertNotIsInstance(frame21.columns, pd.MultiIndex)
        self.assertEqual(frame21.index.shape, (20, ))
        self.assertEqual(frame21.loc["AAPL"].index.shape, (10, ))
        self.assertEqual(frame21.loc["MSFT"].index.shape, (10, ))

        frame22 = qd.load([["AAPL", "MSFT"], ["TSLA", "NIO"]])
        self.assertEqual(frame22.columns.to_list(), [(0, 'Close'), (1, 'Close')])
        self.assertEqual(frame22.index.shape, (20,))
        self.assertEqual(frame22.loc["AAPL/MSFT"].index.shape, (10,))
        self.assertEqual(frame22.loc["TSLA/NIO"].index.shape, (10,))

    def test_age_warning(self):
        qd = QuantData(plugins={"mock": TestQuantData.MokedDataProvider()})

        with self.assertLogs(level='WARN') as cm:
            qd.load("A")
            self.assertIn('is older then', cm.output[0])

    def test_ambiguity(self):
        qd = QuantData(plugins={"mock1": TestQuantData.MokedDataProvider(), "mock2": TestQuantData.MokedDataProvider(2)})

        with self.assertLogs(level='WARN') as cm:
            qd.load("A")
            self.assertIn('Ambiguous', cm.output[0])
            self.assertIn('older then', cm.output[1])

        with self.assertLogs(level='WARN') as cm:
            df = qd.load("A", force_provider='mock2')
            self.assertEqual(df["Close"].mean(), 2)
            self.assertEqual(len(cm.output), 1)

    def test_same_symbol_multiple_providers(self):
        qd = QuantData(plugins={"mock1": TestQuantData.MokedDataProvider(), "mock2": TestQuantData.MokedDataProvider(2)})

        df = qd.load("A|mock1", "A|mock2")
        self.assertIsInstance(df.columns, pd.MultiIndex)
        self.assertEqual(set(df.columns.get_level_values(0)), {'A|mock2', 'A|mock1'})
        self.assertNotIsInstance(df.index, pd.MultiIndex)
