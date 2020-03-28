from unittest import TestCase

from pandas_ml_quant.portfolio.transactionlog import TransactionLog
from pandas_ml_quant_test.config import DF_TEST


class TestBalance(TestCase):

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

        values = trans.evaluate(df["Price"], slippage= lambda _: -2)
        print(values)

    def test_balance_hf_open_close(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)
        trans = TransactionLog()

        # long
        for i in range(2, 8, 2):
            trans.add_open_transaction(i, 10)
            trans.add_close_transaction(i + 1, -10)

        values = trans.evaluate(df["Price"])
        print(values)

    def test_balance_valuation_accumulate(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = 10

        trans = TransactionLog()
        trans.add_open_transaction(2, 10)
        trans.add_open_transaction(4, 2)
        trans.add_close_transaction(5, -12)

        values = trans.evaluate(df["Price"])
        print(values)

    def test_balance_valuation_distribute(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = 10

        trans = TransactionLog()
        trans.add_open_transaction(2, 12)
        trans.add_close_transaction(4, -2)
        trans.add_close_transaction(5, -10)

        values = trans.evaluate(df["Price"])
        print(values)

    def test_long_buy_sell(self):
        df = DF_TEST[-10:].copy()
        df["Price"] = range(1, 11)

        assets = TransactionLog()
        cash = TransactionLog()

        assets.add_open_transaction(2, 10)
        cash.add_open_transaction(2, -10)

        assets.add_close_transaction(4, -10)
        cash.add_close_transaction(4, 10)

        print(assets.evaluate(df["Price"]))
        print(cash.evaluate(df["Price"]))
