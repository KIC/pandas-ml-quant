from unittest import TestCase

import numpy as np

from pandas_ml_quant.trading.transaction_log import TransactionLog, StreamingTransactionLog
from pandas_ml_quant_test.config import DF_TEST


class TestTransactionLog(TestCase):

    def test_balance_valuation_open_close(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)
        trans = TransactionLog()

        # long
        trans.add_open_transaction(2, 10)
        trans.add_close_transaction(4, -10)

        # short
        trans.add_open_transaction(7, -10)
        trans.add_close_transaction(8, 10)

        values = trans.evaluate(df["Price"], slippage=lambda _: -2)
        print(values)
        self.assertEqual(2, values.iloc[-1, -1])

    def test_buy_sell_same_day(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)
        trans = TransactionLog()

        # long
        trans.add_open_transaction(2, 10)
        trans.add_close_transaction(2, -10)

        trans.add_open_transaction(4, 10)
        trans.add_close_transaction(5, -5) # +10
        trans.add_close_transaction(6, -5) # + 5

        values = trans.evaluate(df["Price"])
        print(values)
        self.assertEqual(15, values.iloc[-1, -1])

    def test_open_close_same_day(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)
        trans = TransactionLog()

        # long
        trans.add_open_transaction(2, 10)
        trans.add_close_transaction(2, -2)
        trans.add_open_transaction(3, 2)
        trans.add_close_transaction(3, -1)
        trans.add_close_transaction(4, -9)

        values = trans.evaluate(df["Price"])
        print(values)

        self.assertListEqual([0, 0, -24, -24 -4,  17,  17,  17,  17,  17,  17], values["cash_net"].values.tolist())
        self.assertListEqual([0, 0, +24, +32 +4,   0,   0,   0,   0,   0,   0], values["asset_value"].values.tolist())
        self.assertListEqual([0, 0,   0,      8,  17,  17,  17,  17,  17,  17], values["net"].values.tolist())

    def test_balance_hf_open_close(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)
        trans = TransactionLog()

        # long 3 trades
        for i in range(2, 8, 2):
            trans.add_open_transaction(i, 10)
            trans.add_close_transaction(i + 1, -10)

        values = trans.evaluate(df["Price"])
        print(values)
        self.assertEqual(10 * 3, values.iloc[-1, -1])

    def test_balance_valuation_accumulate(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = 10

        trans = TransactionLog()
        trans.add_open_transaction(2, 10)
        trans.add_open_transaction(4, 2)
        trans.add_close_transaction(5, -12)

        values = trans.evaluate(df["Price"])
        print(values)

    def test_rebalance(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)

        trans = StreamingTransactionLog()
        trans.rebalance(0.1)
        trans.rebalance(0)
        trans.rebalance(0.5)
        trans.rebalance(0.6)
        trans.rebalance(0.4)
        trans.rebalance(0.2)
        trans.rebalance(-0.2)
        trans.rebalance(-0.3)
        trans.rebalance(-0.1)
        trans.rebalance(0)

        values = trans.evaluate(df["Price"])
        print(values)
        self.assertEqual(0, values.iloc[-1, 1])
        np.testing.assert_almost_equal(1.2, values.iloc[-1, -1])
