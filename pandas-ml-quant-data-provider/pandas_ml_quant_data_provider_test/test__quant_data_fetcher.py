from unittest import TestCase

import numpy as np
import pandas as pd

from pandas_ml_quant_data_provider import QuantDataFetcher


def load(symbol: str):
    return pd.DataFrame({"Close": np.ones(10)}, index=pd.date_range('2020-01-01', '2020-01-10'))


class TestQuantData(TestCase):

    def test_shape(self):
        qd = QuantDataFetcher({str: load})

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
        qd = QuantDataFetcher({str: load})

        frame12 = qd.fetch("AAPL", "MSFT")
        self.assertIsInstance(frame12.columns, pd.MultiIndex)
        self.assertNotIsInstance(frame12.index, pd.MultiIndex)
        self.assertListEqual(frame12.columns.to_list(), [("AAPL", "Close"), ("MSFT", "Close")])

        frame21 = qd.fetch([["AAPL", "MSFT"]])
        self.assertIsInstance(frame21.index, pd.MultiIndex)
        self.assertNotIsInstance(frame21.columns, pd.MultiIndex)
        self.assertEqual(frame21.index.shape, (20, ))
        self.assertEqual(frame21.loc["AAPL"].index.shape, (10, ))
        self.assertEqual(frame21.loc["MSFT"].index.shape, (10, ))

        frame22 = qd.fetch([["AAPL", "MSFT"], ["TSLA", "NIO"]])
        self.assertEqual(frame22.columns.to_list(), [(0, 'Close'), (1, 'Close')])
        self.assertEqual(frame22.index.shape, (20,))
        self.assertEqual(frame22.loc["AAPL/MSFT"].index.shape, (10,))
        self.assertEqual(frame22.loc["TSLA/NIO"].index.shape, (10,))
